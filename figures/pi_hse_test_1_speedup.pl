#!/usr/bin/perl -w

my($max_init_time) = 3.0;
my($datafile) = "pi_hse_test_1.csv";

open my $data, $datafile or die "Could not open $datafile: $!";

my($serial_time) = 0;
while (my $line = <$data>)
{   
	my($first_line) = ($. == 1);
	$line =~ s/(?<SIZE>\d+), (?<N>\d+), (?<MPI_INIT>[^\,]+), (?<COMPUTE>[^\,]+), (?<MPI_REDUCE>[^\,]+), (?<TOTAL>[^\,]+)//g;
	my($size) = $+{SIZE};
	my($n) = $+{N};
	my($mpi_init) = $+{MPI_INIT};
	my($compute) = $+{COMPUTE};
	my($mpi_reduce) = $+{MPI_REDUCE};
	my($total) = $+{TOTAL};
	if ($first_line)
	{
		$serial_time = $total;
	}
	my($alpha) = $max_init_time / $serial_time;
	print "$size, $n, " . ($serial_time / $total) . ", " . (1.0 / ($alpha + (1.0 - $alpha) / $n)) . "\n";
}

close($data);
