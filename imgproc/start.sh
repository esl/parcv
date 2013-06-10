erl -pa ebin/ -eval \
'
ok = application:start(sasl),
ok = application:start(imgproc),
ok.
'
