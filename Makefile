.PHONY: install uninstall clean

install:
	pip3 install . -r requirements.txt

uninstall:
	pip3 uninstall token-cutter

clean:
	rm -r build token_cutter.egg-info
