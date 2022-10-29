base='benchmark_fig1'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt

mkdir -p $base
for openmp in "no-"; do
	for m in 1 2 4 8 16 32 64 128; do
		for i in {1..3}; do
			args="--resultsfolder $resultsfolder \
				--codefolder $codefolder-$i \
				--M $m \
				--duration 1\
				--${openmp}openmp"

			cmd="python brunelhakim_M_joined.py --devicename cuda_standalone --no-profiling  \
																					--no-monitors $args"
			echo $cmd
			$cmd 2>&1 | tee -a $logfile
		done
	done
done