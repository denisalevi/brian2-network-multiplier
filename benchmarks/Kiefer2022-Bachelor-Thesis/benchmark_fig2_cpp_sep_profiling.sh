base='benchmark_fig2'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt

mkdir -p $base
for monitors in ""; do
	for openmp in "no-" ""; do
		for i in {1..3}; do
			for j in {1..3}; do
				args="--resultsfolder $resultsfolder \
					--codefolder $codefolder-$j \
					--M 1 \
					--duration 1\
					--${openmp}openmp"

				cmd="python brunelhakim_M_separate.py --devicename cpp_standalone --profiling  \
																							--${monitors}monitors $args"
				echo $cmd
				$cmd 2>&1 | tee -a $logfile
			done
		done
	done
done