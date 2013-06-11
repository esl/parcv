compile:
	gcc -c imgproc_nif.c -o imgproc_nif.o -g -Wall -fPIC -I/usr/lib/erlang/erts-5.10.1/include/ -I/usr/local/browndeer/include/
	gcc -c lodepng.c -o lodepng.o -Wall -fPIC
	gcc -c imgproc.c -o imgproc.o -Wall -fPIC
	gcc imgproc_nif.o lodepng.o -o ../priv/imgproc_nif.so -shared -Wall -fPIC -L/usr/local/lib/ -L/usr/local/browndeer/lib/ -lcoprthr-e -lm -I/usr/lib/erlang/erts-5.10.1/include/ -I/usr/local/browndeer/include/

