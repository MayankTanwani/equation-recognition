def decode_sym(str):
	str = int(str)
	symbols = {
		10 : '+',
		11 : '-',
		12 : '*',
		13 : '/',
	}
	return symbols.get(str,str)