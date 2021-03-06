set terminal png enhanced notransparent
set output 'pi_hse_test_1.png'
set title 'PI MPI+AVX512 Scaling n = 34359738368 (HSE cHARISMa)'
set xlabel 'nprocs'
set ylabel 'time, sec'
plot 'pi_hse_test_1.csv' using 2:3 with linespoints title 'MPI\_Init', 'pi_hse_test_1.csv' using 2:4 with linespoints title 'Integral', 'pi_hse_test_1.csv' using 2:5 with linespoints title 'MPI\_Reduce', 'pi_hse_test_1.csv' using 2:6 with linespoints title 'total'

