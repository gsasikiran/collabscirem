# Models evaluated in the paper, as <name>=<HF id or merged model dir>.
# Sourced by the experiments/*/run.sh scripts. <name> is the vLLM served name (MINT) and
# the output directory name (single-turn); fine-tuned models are written by
# experiments/02_training/train.sh to outputs/merged/<name>.

MAIN_MODELS=(
    Llama-3.2-3B-Instruct=meta-llama/Llama-3.2-3B-Instruct
    llama-3.2-3B-collabllmsci=outputs/merged/llama-3.2-3B-collabllmsci
    llama-3.2-3B-collabscirem=outputs/merged/llama-3.2-3B-collabscirem
    Llama-3.1-8B-Instruct=meta-llama/Llama-3.1-8B-Instruct
    llama-3.1-8B-collabllmsci=outputs/merged/llama-3.1-8B-collabllmsci
    llama-3.1-8B-collabscirem=outputs/merged/llama-3.1-8B-collabscirem
    Qwen2.5-3B-Instruct=Qwen/Qwen2.5-3B-Instruct
    qwen-2.5-3B-collabllmsci=outputs/merged/qwen-2.5-3B-collabllmsci
    qwen-2.5-3B-collabscirem=outputs/merged/qwen-2.5-3B-collabscirem
    Qwen2.5-7B-Instruct=Qwen/Qwen2.5-7B-Instruct
    qwen-2.5-7B-collabllmsci=outputs/merged/qwen-2.5-7B-collabllmsci
    qwen-2.5-7B-collabscirem=outputs/merged/qwen-2.5-7B-collabscirem
)

# Llama-3.1-8B ablations and sensitivity (Table 2)
ABLATION_MODELS=(
    llama-3.1-8B-ablation-gen_struct=outputs/merged/llama-3.1-8B-ablation-gen_struct
    llama-3.1-8B-ablation-gen_sci=outputs/merged/llama-3.1-8B-ablation-gen_sci
    llama-3.1-8B-ablation-struct_sci=outputs/merged/llama-3.1-8B-ablation-struct_sci
    llama-3.1-8B-collabllm-sciw=outputs/merged/llama-3.1-8B-collabllm-sciw
    llama-3.1-8B-sensitivity=outputs/merged/llama-3.1-8B-sensitivity
)

# MODELS=main|ablation|all selects the list (default: all)
select_models() {
    case "${MODELS:-all}" in
        main) SELECTED=("${MAIN_MODELS[@]}") ;;
        ablation) SELECTED=("${ABLATION_MODELS[@]}") ;;
        all) SELECTED=("${MAIN_MODELS[@]}" "${ABLATION_MODELS[@]}") ;;
        *) echo "MODELS must be main, ablation or all" >&2; return 1 ;;
    esac
}
