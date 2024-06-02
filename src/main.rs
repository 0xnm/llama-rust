use std::{env, io, process};
use std::fs::File;
use std::io::Read;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    // default parameters
    let mut checkpoint_path;  // e.g. out/model.bin
    let mut tokenizer_path = "tokenizer.bin";
    let mut temperature = 1f32;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    let mut topp = 0.9f32;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let mut steps = 256;            // number of steps to run for
    let mut prompt = "";        // prompt string
    let mut rng_seed = 0u64; // seed rng with time by default
    let mut mode = "generate";    // generate|chat
    let mut system_prompt= Option::<String>::None; // the (optional) system prompt to use in chat mode

    let args: Vec<String> = env::args().collect();
    let argc = args.len();
    // poor man's C argparse, so we can override the defaults above from the command line
    if args.len() >= 2 { checkpoint_path = &args[1]; } else { error_usage(); }
    for i in (2..argc).step_by(2) {
        // do some basic validation
        if i + 1 >= argc { error_usage(); } // must have arg after flag
        if args[i][0] != '-' { error_usage(); } // must start with dash
        if args[i].len() != 2 { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if args[i][1] == 't' {
            temperature = f32::from_str(args[i + 1].as_str()).unwrap();
        } else if args[i][1] == 'p' {
            topp = f32::from_str(args[i + 1].as_str()).unwrap();
        } else if args[i][1] == 's' {
            rng_seed = u64::from_str(args[i + 1].as_str()).unwrap();
        } else if args[i][1] == 'n' {
            steps = i32::from_str(args[i + 1].as_str()).unwrap();
        } else if args[i][1] == 'i' {
            prompt = &args[i + 1];
        } else if args[i][1] == 'z' {
            tokenizer_path = &args[i + 1];
        } else if args[i][1] == 'm' {
            mode = &args[i + 1];
        } else if args[i][1] == 'y' {
            system_prompt = Some(args[i + 1].to_owned());
        } else {
            error_usage();
        }
    }

    // parameter validation/overrides
    if rng_seed <= 0 { rng_seed = Instant::now().elapsed().as_millis().into(); }
    if temperature < 0.0 { temperature = 0.0; }
    if topp < 0.0 || 1.0 < topp { topp = 0.9; }
    if steps < 0 { steps = 0; }

    // build the Transformer via the model .bin file
    let transformer = build_transformer(checkpoint_path)?;
    if steps == 0 || steps > transformer.config.seq_len { steps = transformer.config.seq_len; } // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    let tokenizer = build_tokenizer(tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    let sampler = build_sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if mode == "generate" {
        generate(transformer, tokenizer, sampler, prompt, steps);
    } else if mode == "chat" {
        chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
    } else {
        eprintln!("unknown mode: {}", mode);
        error_usage();
    }
}

/* Inference for Llama-2 Transformer model in pure Rust */

// ----------------------------------------------------------------------------
// Transformer model

#[repr(C, packed)]
struct Config {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>,    // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wo: Vec<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Vec<f32>,
}

struct RunState {
    // current wave of activations
    x: Vec<f32>, // activation at current time stamp (dim,)
    xb: Vec<f32>, // same, but inside a residual branch (dim,)
    xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
    hb: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>, // query (dim,)
    k: Vec<f32>, // key (dim,)
    v: Vec<f32>, // value (dim,)
    att: Vec<f32>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

struct Transformer {
    config: Config, // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState, // buffers for the "wave" of activations in the forward pass
}
fn memory_map_weights(config: &Config, raw_weights: &[u8], shared_weights: bool) -> TransformerWeights {
    let head_size = config.dim / config.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    let n_layers = config.n_layers;
    let mut index = 0usize;
    let mut next = (config.vocab_size * config.dim * 4) as usize;
    let token_embedding_table = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * 4) as usize;
    let rms_att_weight = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * (config.n_heads * head_size) * 4) as usize;
    let wq = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * (config.n_kv_heads * head_size) * 4) as usize;
    let wk = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * (config.n_kv_heads * head_size) * 4) as usize;
    let wv = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * (config.n_heads * head_size) * config.dim * 4) as usize;
    let wo = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * 4) as usize;
    let rms_ffn_weight = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * config.hidden_dim * 4) as usize;
    let w1 = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.hidden_dim * config.dim * 4) as usize;
    let w2 = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (n_layers * config.dim * config.hidden_dim * 4) as usize;
    let w3 = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    next += (config.dim * 4) as usize;
    let rms_final_weight = unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..next]) };
    index = next;
    index += ((config.seq_len * head_size / 2) * 4) as usize; // skip what used to be freq_cis_real (for RoPE)
    index += ((config.seq_len * head_size / 2) * 4) as usize; // skip what used to be freq_cis_imag (for RoPE)
    let wcls = if shared_weights { token_embedding_table.clone() } else { unsafe { std::mem::transmute::<[u8], Vec<f32>>(raw_weights[index..index + 1000]) } };
    TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        wq,
        wk,
        wv,
        wo,
        rms_ffn_weight,
        w1,
        w2,
        w3,
        rms_final_weight,
        wcls,
    }
}

fn read_checkpoint(checkpoint: &str) -> Result<(Config, TransformerWeights), ()> {
    let mut file = File::open(checkpoint).expect(&format!("Couldn't open file {}", checkpoint));
// read in the config header
    let mut config_buf = [0u8; std::mem::size_of::<Config>()];
    file.read_exact(config_buf.as_mut_slice())?;
    let mut config = unsafe {
        std::mem::transmute::<&[u8], Config>(config_buf)
    };
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = config.vocab_size > 0;
    config.vocab_size = config.vocab_size.abs();
    let mut raw_weights_buf = Vec::<u8>::new();
    file.read_to_end(&mut raw_weights_buf)?;
    let weights = memory_map_weights(&config, raw_weights_buf.as_slice(), shared_weights);
    Ok((config, weights))
}

fn build_transformer(checkpoint_path: &str) -> Result<Transformer, ()> {
    // read in the Config and the Weights from the checkpoint
    let (config, weights) = read_checkpoint(checkpoint_path)?;

    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    let x = Vec::<f32>::with_capacity(config.dim.into());
    let xb = Vec::<f32>::with_capacity(config.dim.into());
    let xb2 = Vec::<f32>::with_capacity(config.dim.into());
    let hb = Vec::<f32>::with_capacity(config.hidden_dim.into());
    let hb2 = Vec::<f32>::with_capacity(config.hidden_dim.into());
    let q = Vec::<f32>::with_capacity(config.dim.into());
    let key_cache = Vec::<f32>::with_capacity((config.n_layers * config.seq_len * kv_dim).into());
    let value_cache =  Vec::<f32>::with_capacity((config.n_layers * config.seq_len * kv_dim).into());
    let att =        Vec::<f32>::with_capacity((config.n_heads * config.seq_len).into());
    let logits = Vec::<f32>::with_capacity(config.vocab_size.into());
    let run_state = RunState {
        x,
        xb,
        xb2,
        hb,
        hb2,
        q,
        key_cache,
        value_cache,
        att,
        logits
    };
    Ok(Transformer {
        config,
        weights,
        state: run_state,
    })
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: i32) {
    // calculate sum of squares
    let mut ss = 0f32;
    for j in 0..size as usize {
        ss += x[j] * x[j];
    }
    ss = ss / size as f32;
    ss += 1e-5f32;
    ss = 1f32 / ss.sqrt();
    // normalize and scale
    for j in 0..size as usize {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn softmax(x: &mut [f32], size: i32) {
    // find max value (for numerical stability)
    let mut max_val = x[0];
    for i in 1..size as usize {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    // exp and sum
    let mut sum = 0f32;
    for i in 0..size as usize {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    // normalize
    for i in 0..size as usize {
        x[i] /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: i32, d: i32) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for i in 0..d {
        let mut val = 0f32;
        for j in 0..n {
            val += w[(i * n + j) as usize] * x[j as usize];
        }
        xout[i as usize] = val;
    }
}

fn forward(transformer: &mut Transformer, token: i32, pos: i32) -> Vec<f32> {

    // a few convenience variables
    let config = &transformer.config;
    let weights = &transformer.weights;
    let state = &mut transformer.state;
    let dim = config.dim;
    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    let kv_mul = config.n_heads / config.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    let hidden_dim = config.hidden_dim;
    let head_size = dim / config.n_heads;

    // copy the token embedding into x
    state.x = weights.token_embedding_table[(token*dim) as usize..].iter().collect();
    let x = &state.x;

    // forward all the layers
    for l in 0..config.n_layers {
        // attention rmsnorm
        rmsnorm(&mut state.xb, x, &weights.rms_att_weight[(l * dim) as usize..], dim);

        // key and value point to the kv cache
        let loff = l * config.seq_len * kv_dim; // kv cache layer offset for convenience
        state.k = state.key_cache[loff + pos * kv_dim..];
        state.v = state.value_cache[loff + pos * kv_dim..];

        // qkv matmuls for this position
        matmul(&mut state.q, &state.xb, &weights.wq[(l * dim * dim) as usize..], dim, dim);
        matmul(&mut state.k, &state.xb, &weights.wk[(l * dim * kv_dim) as usize..], dim, kv_dim);
        matmul(&mut state.v, &state.xb, &weights.wv[(l * dim * kv_dim) as usize..], dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in (0..dim as usize).step_by(2) {
            let head_dim = i as i32 % head_size;
            let freq = 1f32 / 10000f32.powf(head_dim as f32 / (head_size as f32));
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim  { 2 } else {1 }; // how many vectors? 2 = q & k, 1 = q only
            for v in 0..rotn {
                let vec = if v == 0  { &mut state.q }  else { &mut state.k }; // the vector to rotate (query or key)
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        for h in 0..config.n_heads {
            // get the query vector for this head
            let q = &state.q[(h * head_size) as usize..];
            // attention scores for this head
            let att = state.att[(h * config.seq_len) as usize..];
            // iterate over all timesteps, including the current one
            for t in 0..pos+1 {
                // get the key vector for this head and at this timestep
                let k = state.key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size) as usize..];
                // calculate the attention score as the dot product of q and k
                let mut score = 0f32;
                for i in 0..head_size as usize {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            let xb = state.xb[(h * head_size) as usize..];
            memset(xb, 0, head_size * sizeof(float));
            for t in 0..pos+1 {
                // get the value vector for this head and at this timestep
                let v = state.value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size) as usize..];
                // get the attention weight for this timestep
                let a = att[t];
                // accumulate the weighted value into xb
                for i in 0..head_size as usize {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(&mut state.xb2, &state.xb, &weights.wo[(l * dim * dim) as usize..], dim, dim);

        // residual connection back into x
        for i in 0..dim as usize {
            x[i] += state.xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(&mut state.xb, x, &weights.rms_ffn_weight[(l * dim) as usize..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut state.hb, &state.xb, &weights.w1[(l * dim * hidden_dim) as usize..], dim, hidden_dim);
        matmul(&mut state.hb2, &state.xb, &weights.w3[(l * dim * hidden_dim) as usize..], dim, hidden_dim);

        // SwiGLU non-linearity
        for i in 0..hidden_dim as usize {
            let mut val = state.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= 1f32 / (1f32 + (-val).exp());
            // elementwise multiply with w3(x)
            val *= state.hb2[i];
            state.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(&mut state.xb, &state.hb, &weights.w2[(l * dim * hidden_dim) as usize..], hidden_dim, dim);

        // residual connection
        for i in 0..dim as usize {
            x[i] += state.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(&mut state.x, x, &weights.rms_final_weight, dim);

    // classifier into logits
    matmul(&mut state.logits, x, &weights.wcls, config.dim, config.vocab_size);
    return state.logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct TokenIndex {
    str: String,
    id: i32,
}

struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    sorted_vocab: Option<Vec<TokenIndex>>,
    vocab_size: i32,
    max_token_length: u32,
    byte_pieces: [char; 512], // stores all single-byte strings
}

fn compare_tokens(a: &TokenIndex, b: &TokenIndex) -> i32 {
    return a.str.cmp(&b.str).into();
}

fn build_tokenizer(tokenizer_path: &str, vocab_size: i32) -> Tokenizer {
    let mut vocab = Vec::<String>::new();
    let mut vocab_scores = Vec::<f32>::with_capacity(vocab_size.into());
    let mut byte_pieces = ['\0'; 512];
    for i in 0..256 {
        byte_pieces[i * 2] = i.into();
    }
    // read in the file
    let mut file = File::open(tokenizer_path).expect(&format!("couldn't load {}", tokenizer_path));
    let mut buf = [0u8;4];
    file.read_exact(&mut buf).expect("failed read max token length");
    let max_token_length = unsafe {
      std::mem::transmute::<[u8], u32>(buf)
    };
    for _ in 0..vocab_size {
        file.read_exact(&mut buf).expect("failed to read vocab_score");
        let vocab_score = unsafe {
            std::mem::transmute::<[u8], f32>(buf)
        };
        vocab_scores.push(vocab_score);
        file.read_exact(&mut buf).expect("failed to read vocab word length");
        let word_length = unsafe {
            std::mem::transmute::<[u8], u32>(buf)
        };
        let mut buf = Vec::with_capacity(word_length as usize);
        file.read_exact(&mut buf).expect("failed to read vocab word");
        let word = String::from_utf8(buf).expect("failed to convert vocab word to utf8");
        vocab.push(word)
    }
    Tokenizer {
        byte_pieces,
        max_token_length,
        sorted_vocab: None,
        vocab_scores,
        vocab,
        vocab_size
    }
}

fn decode(t: &Tokenizer, prev_token: i32, token: usize) -> String {
    let piece = &t.vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if prev_token == 1 && piece.chars()[0] == ' ' { piece += 1; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    let byte_val;
    if sscanf(piece, "<0x%02hhX>", &byte_val) == 1 {
        piece = (char *) t.byte_pieces + byte_val * 2;
    }
    return piece;
}

fn safe_printf(piece: &str) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    let chars = piece.chars();
    if chars[0] == '\0' { return; }
    if chars[1] == '\0' {
        let byte_val = chars[0];
        if !(isprint(byte_val) || isspace(byte_val)) {
            return; // bad byte, don't print it
        }
    }
    print!("{}", piece);
}

fn str_lookup(str: &str, sorted_vocab: &[TokenIndex]) -> Option<i32> {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    let tok = TokenIndex { str: str.into_string(), id: -1 }; // acts as the key to search for
    let result = sorted_vocab.binary_search_by(|probe| compare_tokens(&tok, probe).into());
    return match result {
        Ok(index) => Some(sorted_vocab[index].id),
        Err(_) => None
    };
}

fn encode(t: &mut Tokenizer, text: &str, bos: bool, eos: bool, tokens: &mut [i32]) -> i32 {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if t.sorted_vocab.is_none() {
        // lazily malloc and sort the vocabulary
        let mut sorted_vocab = Vec::<TokenIndex>::with_capacity(t.vocab_size as usize);
        for i in 0..t.vocab_size as usize {
            sorted_vocab[i] = TokenIndex {
                str: t.vocab[i].clone(),
                id: i as i32
            };
        }
        sorted_vocab.sort_by(compare_tokens);
        t.sorted_vocab = Some(sorted_vocab)
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    let mut str_buffer = Vec::<char>::with_capacity(t.max_token_length as usize * 2 + 1 + 2);
    let mut str_len = 0;

    // start at 0 tokens
    let mut n_tokens = 0i32;

    // add optional BOS (=1) token, if desired
    if bos {
        tokens[n_tokens] = 1;
        n_tokens += 1;
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if text.chars()[0] != '\0' {
        let dummy_prefix = str_lookup(" ", t.sorted_vocab.as_ref().unwrap()).unwrap();
        tokens[n_tokens] = dummy_prefix;
        n_tokens += 1;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx
    
    // process the raw (UTF-8) byte sequence of the input string
    let chars = text.chars().collect::<Vec<&char>>();
    for c in chars {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if (*c as u8 & 0xC0) != 0x80 {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len] = *c; // ++ is post-increment, incremented after this line
        str_len += 1;
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if (*(c + 1) & 0xC0) == 0x80 && str_len < 4 {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        let id = str_lookup(&str_buffer, t.sorted_vocab.as_ref().unwrap());

        if id.is_some() {
            // we found this codepoint in vocab, add it as a token
            tokens[n_tokens] = id.unwrap();
            n_tokens += 1;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for i in 0..str_len {
                tokens[n_tokens] = str_buffer[i] as i32 + 3;
                n_tokens += 1;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    loop {
        let mut best_score = -1e10;
        let mut best_id = -1;
        let mut best_idx = -1;

        for i in 0..n_tokens-1 {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            str_buffer = format!("{}{}", t.vocab[tokens[i]], t.vocab[tokens[i + 1]]);
            let id = str_lookup(str_buffer, t.sorted_vocab.as_ref().unwrap());
            if id != -1 && t.vocab_scores[id] > best_score {
                // this merge pair exists in vocab! record its score and position
                best_score = t.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if best_idx == -1 {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for i in best_idx + 1..n_tokens-1 {
            tokens[i] = tokens[i + 1];
        }
        n_tokens -= 1; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if eos {
        tokens[n_tokens] = 2;
        n_tokens+=1;
    }

    return n_tokens
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct ProbIndex {
    prob: f32,
    index: i32,
} // struct used when sorting probabilities during top-p sampling

struct Sampler {
    vocab_size: i32,
    prob_index: Vec<ProbIndex>, // buffer used in top-p sampling
    temperature: f32,
    topp: f32,
    rng_state: u64,
}

fn sample_argmax(probabilities: &[f32], n: i32) -> i32 {
// return the index that has the highest probability
    let mut max_i = 0;
    let mut max_p = probabilities[0];
    for i in 1..n {
        if probabilities[i] > max_p {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

fn sample_mult(probabilities: &[f32], n: i32, coin: f32) -> i32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if coin < cdf {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

fn compare(a: &ProbIndex, b: &ProbIndex) -> i32 {
    return if a.prob > b.prob { -1 } else if a.prob < b.prob { 1 } else { 0 };
}

fn sample_topp(probabilities: &[f32], n: i32, topp: f32, prob_index: &mut Vec<ProbIndex>, coin: f32) -> i32 {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    let mut n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    let cutoff = (1f32 - topp) / (n - 1) as f32;
    for i in 0..n {
        if probabilities[i] >= cutoff {
            prob_index[n0].index = i;
            prob_index[n0].prob = probabilities[i];
            n0 += 1;
        }
    }
    prob_index[0..n0].sort_by(compare);

    // truncate the list where cumulative probability exceeds topp
    let mut cumulative_prob = 0f32;
    let mut last_idx = n0 - 1; // in case of rounding errors consider all elements
    for i in 0..n0 {
        cumulative_prob += prob_index[i].prob;
        if cumulative_prob > topp {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    let r = coin * cumulative_prob;
    let mut cdf = 0f32;
    for i in 0..last_idx+1 {
        cdf += prob_index[i].prob;
        if r < cdf {
            return prob_index[i].index;
        }
    }
    return prob_index[last_idx].index; // in case of rounding errors
}

fn build_sampler(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Sampler {
    return Sampler {
        vocab_size,
        temperature,
        topp,
        rng_state: rng_seed,
        // buffer only used with nucleus sampling; may not need but it's ~small
        prob_index: Vec::<ProbIndex>::with_capacity(vocab_size as usize)
    };
}

fn random_u32(state: &mut u64) -> u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return ((*state * 0x2545F4914F6CDD1Du64) >> 32) as u32;
}

fn random_f32(state: &mut u64) -> f32 { // random float32 in [0,1)
    return (random_u32(state) >> 8) as f32 / 16777216f32;
}

fn sample(sampler: &mut Sampler, logits: &mut [f32]) -> i32 {
    // sample the token given the logits and some hyperparameters
    let next;
    if sampler.temperature == 0f32 {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler.vocab_size);
    } else {
        // apply the temperature to the logits
        for q in 0..sampler.vocab_size { logits[q] /= sampler.temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler.vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        let coin = random_f32(&mut sampler.rng_state);
        // we sample from this distribution to get the next token
        if sampler.topp <= 0f32 || sampler.topp >= 1f32 {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler.vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler.vocab_size, sampler.topp, &mut sampler.prob_index, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

fn time_in_ms() -> u64 {
    // return time in milliseconds, for benchmarking the model speed
    return Instant::now().elapsed().as_millis().into();
}

// ----------------------------------------------------------------------------
// generation loop

fn generate(mut transformer: Transformer, mut tokenizer: Tokenizer, mut sampler: Sampler, prompt: &str, steps: i32) {
    // encode the (string) prompt into tokens sequence
    let mut prompt_tokens = Vec::<i32>::with_capacity (prompt.len() + 3); // +3 for '\0', ?BOS, ?EOS
    let num_prompt_tokens = encode(&mut tokenizer, prompt, true, false, &mut prompt_tokens);
    if num_prompt_tokens < 1 {
        eprintln!("something is wrong, expected at least 1 prompt token");
        process::exit(1);
    }

    // start the main loop
    let mut start = 0;  // used to time our code, only initialized after first iteration
    let next;        // will store the next token in the sequence
    let token = prompt_tokens[0]; // kick off with the first token in the prompt
    let mut pos = 0;     // position in the sequence
    while pos < steps {
        // forward the transformer to get logits for the next token
        let mut logits = forward(&mut transformer, token, pos);

        // advance the state machine
        if pos < num_prompt_tokens - 1 {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(&mut sampler, &mut logits);
        }
        pos+=1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 { break; }

        // print the token as string, decode it with the Tokenizer object
        let piece = decode(&tokenizer, token, next);
        safe_printf(&piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;

        // init the timer here because the first iteration can be slower
        if start == 0 { start = time_in_ms(); }
    }
    println!();

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = time_in_ms();
        eprintln!("achieved tok/s: {}", (pos - 1) as f32 / ((end - start) * 1000) as f32);
    }
}

fn read_stdin(guide: &str) -> String {
    // read a line from stdin, up to but not including \n
    println!("{}", guide);
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf);
    buf
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

fn chat(mut transformer: Transformer, mut tokenizer: Tokenizer, mut sampler: Sampler,
        cli_user_prompt: Option<String>, cli_system_prompt: Option<String>, steps: i32) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are somewhat haphazardly and unsafely set atm
    let mut system_prompt: String;
    let mut user_prompt: String;
    let mut rendered_prompt: String;
    let mut num_prompt_tokens = 0;
    let mut prompt_tokens = [0i32; 1152];
    let mut user_idx = 0;

    // start the main loop
    let mut user_turn = 1; // user starts
    let next;        // will store the next token in the sequence
    let token;       // stores the current token to feed into the transformer
    let mut pos = 0;     // position in the sequence
    while pos < steps {
        // when it is the user's turn to contribute tokens to the dialog...
        if user_turn {
            // get the (optional) system prompt at position 0
            if pos == 0 {
                // at position 0, the user can also contribute a system prompt
                if cli_system_prompt == None {
                    // system prompt was not passed in, attempt to get it from stdin
                    system_prompt = read_stdin("Enter system prompt (optional): ");
                } else {
                    // system prompt was passed in, use it
                    system_prompt = cli_system_prompt.as_ref().unwrap().clone();
                }
            }
            // get the user prompt
            if pos == 0 && cli_user_prompt != None {
                // user prompt for position 0 was passed in, use it
                user_prompt = cli_user_prompt.as_ref().unwrap().clone();
            } else {
                // otherwise get user prompt from stdin
                user_prompt = read_stdin("User: ");
            }
            // render user/system prompts into the Llama 2 Chat schema
            if pos == 0 && system_prompt[0] != '\0' {
                rendered_prompt = format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", system_prompt, user_prompt);
            } else {
                rendered_prompt = format!("[INST] {} [/INST]", user_prompt);
            }
            // encode the rendered prompt into tokens
            num_prompt_tokens = encode(&mut tokenizer, &rendered_prompt, true, false, &mut prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            print!("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if user_idx < num_prompt_tokens {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if token == 2 { user_turn = 1; }

        // forward the transformer to get logits for the next token
        let mut logits = forward(&mut transformer, token, pos);
        next = sample(&mut sampler, &mut logits);
        pos += 1;

        if user_idx >= num_prompt_tokens && next != 2 {
            // the Assistant is responding, so print its output
            let piece = decode(&mut tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        }
        if next == 2 { println!(); }
    }
    println!();
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
//# ifndef TESTING

fn error_usage() {
    eprintln!("Usage:   run <checkpoint> [options]");
    eprintln!("Example: run model.bin -n 256 -i \"Once upon a time\"");
    eprintln!("Options:");
    eprintln!("  -t <float>  temperature in [0,inf], default 1.0");
    eprintln!("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
    eprintln!("  -s <int>    random seed, default time(NULL)");
    eprintln!("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
    eprintln!("  -i <string> input prompt");
    eprintln!("  -z <string> optional path to custom tokenizer");
    eprintln!("  -m <string> mode: generate|chat, default: generate");
    eprintln!("  -y <string> (optional) system prompt in chat mode");
    process::exit(1);
}
//# endif
