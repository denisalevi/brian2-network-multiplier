base='benchmark_fig2'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt

mkdir -p $base
for monitors in ""; do
	for openmp in  "no-"; do
		for m in 1 2 4 8 16 32 64 128; do
			for i in 1; do
				args="--resultsfolder $resultsfolder \
					--codefolder $codefolder-$i \
					--M $m \
					--duration 1\
					--${openmp}openmp"

				cmd="python brunelhakim_M_joined.py --devicename cpp_standalone --profiling  \
																						--${monitors}monitors $args"
				echo $cmd
				$cmd 2>&1 | tee -a $logfile
			done
		done
	done 
done