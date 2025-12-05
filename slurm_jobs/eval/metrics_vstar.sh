LOG_FILE=/your/path/to/log/file
mkdir -p $LOG_FILE


API_URL=

python -u ./eval/vstar/judge_result.py \
    --api_url $API_URL \
    --vstar_bench_path /your/path/to/vstar_bench \
    --model_path /your/path/to/model \
    --save_name inference_vstar \
    --save_dir /your/dir/to/save/result \
    2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' > $LOG_FILE/metrics_vstar.log &
    
wait 
echo "Finish inference_vstar"

