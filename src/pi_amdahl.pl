#!/usr/bin/perl

my($sbatch) = join("", `cat ../slurm.sbatch.in`);
my($size) = 1024 * 1024 * 1024 * 256;

my($n) = 1;
my($n_max) = 2048;
for ( ; $n < $n_max; $n++)
{
	# Substitute values into sbatch file template
	my($sbatch_n) = $sbatch;
	$sbatch_n =~ s/<NTASKS>/$n/g;
	$sbatch_n =~ s/<SIZE>/$size/g;

	open(my $fh, '>', "slurm.sbatch") or die "Cannot open '$pi_csv' $!";
	print $fh "$sbatch_n";
	close($fh);

	my($log) = join("", `sbatch slurm.sbatch`);
	$log =~ s/Submitted batch job (?<JOB>\d+)//g;
	my($job) = $+{JOB};
	sleep(1);

	# Check the job until we get init/reduce/total times
	while (1)
	{
		my($status) = join("", `cat pi_amdahl_$job.log 2>/dev/null`);
		$status =~ s/Init time = (?<INIT_TIME>.*) sec//g;
                if (!defined($+{INIT_TIME}))
                {
                        next;
                }
		my($init_time) = $+{INIT_TIME};
		$status =~ s/Reduction time = (?<REDUCTION_TIME>.*) sec//g;
                if (!defined($+{REDUCTION_TIME}))
                {
                        next;
                }
		my($reduction_time) = $+{REDUCTION_TIME};
		$status =~ s/time = (?<TIME>.*) sec//g;
		if (!defined($+{TIME}))
		{
			next;
		}
		my($time) = $+{TIME};
		
		# Result has been collected, log and proceed to the next one
		print "$size, $n, $init_time, " . ($time - $reduction_time - $init_time) . ", $reduction_time, $time\n";
		last;
	}
}

