gcc -O3 -march=native -I/usr/include -L/usr/lib -o c_bench_compare \
       c_bench_compare.c -lisal -pthread