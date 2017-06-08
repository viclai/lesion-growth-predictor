def ctrls():
	controls = {
		'Help' : 'h',
		'Home' : 'home',
		'Skip' : '',
		'Quit' : 'q'
	}
	return controls

def special_keys():
	special_keys = {
		'' : 'ENTER'
	}
	return special_keys

def print_controls():
	print 'Controls:'
	for name, key in ctrls().iteritems():
		if key in special_keys():
			key = special_keys()[key]
		print '\t' + name + ': [' + key + ']'
	print
