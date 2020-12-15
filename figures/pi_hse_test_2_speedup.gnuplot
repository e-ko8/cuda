set terminal png enhanced notransparent
set output 'pi_hse_test_2_speedup.png'
set title 'PI MPI+AVX512 Scaling n = 274877906944 (HSE cHARISMa)'
set xlabel 'nprocs'
set ylabel 'speedup'
plot 'pi_hse_test_2_speedup.csv' using 2:3 with linespoints title 'actual speedup', 'pi_hse_test_2_speedup.csv' using 2:4 with linespoints title 'Amdahl speedup'
