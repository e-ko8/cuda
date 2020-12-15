#!/usr/bin/perl

my($pi_csv) = "pi.csv";
open(my $fh, '>', $pi_csv) or die "Cannot open '$pi_csv' $!";

my($n) = 1024;
my($n_max) = 1024 * 1024 * 1024;
for ( ; $n <= $n_max; $n *= 2)
{
	my($log_pi_mpi1) = join("", `./pi_mpi_1 $n`);
	$log_pi_mpi1 =~ s/time = (?<TIME1>.*) sec//g;
	if (!defined($+{TIME1}))
	{
		print STDERR "Cound not find the time result\n";
	}
	my($time_pi_mpi1) = $+{TIME1};

        my($log_pi_mpi_openmp2) = join("", `./pi_mpi_openmp_2 $n`);
        $log_pi_mpi_openmp2 =~ s/time = (?<TIME2>.*) sec//g;
        if (!defined($+{TIME2}))
        {
                print STDERR "Cound not find the time result\n";
        }
        my($time_pi_mpi_openmp2) = $+{TIME2};

        my($log_pi_mpi_openmp_xsimd3) = join("", `./pi_mpi_openmp_xsimd_3 $n`);
        $log_pi_mpi_openmp_xsimd3 =~ s/time = (?<TIME3>.*) sec//g;
        if (!defined($+{TIME3}))
        {
                print STDERR "Cound not find the time result\n";
        }
        my($time_pi_mpi_openmp_xsimd3) = $+{TIME3};

	print $fh "$n, $time_pi_mpi1, $time_pi_mpi_openmp2, $time_pi_mpi_openmp_xsimd3\n";
}

close($fh);

my($pi_gnuplot) = "pi.gnuplot";
open(my $fh, '>', $pi_gnuplot) or die "Cannot open '$pi_gnuplot' $!";

my($cpu) = `cat /proc/cpuinfo | grep "model name"`;
$cpu =~ s/model\sname\s*:\s*//g;
$cpu =~ s/\n//g;

print $fh "set terminal png enhanced notransparent\n";
print $fh "set output 'pi.png'\n";
print $fh "set title 'PI Optimizations ($cpu)'\n";
print $fh "set xlabel 'n'\n";
print $fh "set ylabel 'time, sec'\n";
print $fh "plot '$pi_csv' using 1:2 with linespoints title 'MPI', '$pi_csv' using 1:3 with linespoints title 'MPI+OpenMP', '$pi_csv' using 1:4 with linespoints title 'MPI+OpenMP+SIMD'\n";

close($fh);

system("gnuplot $pi_gnuplot");

