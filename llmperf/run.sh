CURRENTDATE=$(date +"%d-%m-%Y-%H-%M-%S")
printf "date is ${CURRENTDATE}"

 
python check.py \
--model "Mistral" \
--mean-input-tokens 100 \
--stddev-input-tokens 2 \
--mean-output-tokens 400 \
--stddev-output-tokens 2 \
--max-num-completed-requests 50 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "aa" \
--llm-api openai \
--additional-sampling-params '{}'
