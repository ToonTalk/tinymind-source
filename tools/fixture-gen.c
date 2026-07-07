/* fixture-gen.c — golden greedy-token generator for E2/G1 parity.
 * Reuses llama2.c's own forward/encode/argmax by including run.c directly, so the
 * fixtures ARE the reference implementation's output, not a reimplementation.
 * Prints exactly `steps` token ids (prompt tokens included), one per line.
 * Deliberately does NOT stop on BOS/EOS: fixed-length sequences make the JS
 * comparison strict. Build: gcc -O2 -Dmain=run_c_main -o fixture-gen fixture-gen.c -lm
 */
#define main run_c_main_unused
#include "../vendor/llama2.c/run.c"
#undef main

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: fixture-gen <checkpoint.bin> <tokenizer.bin> <prompt> <steps>\n");
        return 1;
    }
    char *ckpt = argv[1], *tokpath = argv[2], *prompt = argv[3];
    int steps = atoi(argv[4]);

    Transformer transformer;
    build_transformer(&transformer, ckpt);
    if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokpath, transformer.config.vocab_size);

    int *prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "no prompt tokens\n"); return 1; }

    int token = prompt_tokens[0];
    printf("%d\n", token);
    for (int pos = 0; pos < steps - 1; pos++) {
        float *logits = forward(&transformer, token, pos);
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(logits, transformer.config.vocab_size);
        }
        printf("%d\n", next);
        token = next;
    }
    return 0;
}
