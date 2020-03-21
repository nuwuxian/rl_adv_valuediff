for i in $(seq 0 20); do
	python -m bin.advtrain_ppo --job_name=actor --learner_ip localhost &
done;
