compile:
	gcc -c ocl.c -o ocl.o -Wall -fPIC -shared -g -I/usr/lib/erlang/erts-5.10.1/include/ -I/usr/local/browndeer/include/
	gcc ocl.o -o ../priv/ocl.so -shared -Wall -fPIC -L/usr/local/browndeer/lib/ -lcoprthr-e -I/usr/lib/erlang/erts-5.10.1/include/ -I/usr/local/browndeer/include/
