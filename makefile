
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LIB
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

_clean_c:
	find src/gravitation/kernel/cy* -name '*.c' -exec rm -f {} +

_clean_bin:
	find src/ -name '*.dll' -exec rm -f {} +
	find src/ -name '*.so' -exec rm -f {} +

_clean_octave:
	find src/ -name 'octave-workspace' -exec rm -f {} +

_clean_plot:
	find src/ -name '*.htm' -exec rm -f {} +
	find src/ -name '*.html' -exec rm -f {} +

_clean_py:
	find src/ -name '*.pyc' -exec rm -f {} +
	find src/ -name '*.pyo' -exec rm -f {} +
	find src/ -name '*~' -exec rm -f {} +
	find src/ -name '__pycache__' -exec rm -fr {} +

_clean_release:
	-rm -r build/*
	-rm -r dist/*
	-rm -r pip-wheel-metadata/*

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ENTRY POINTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

black:
	black .

clean:
	make _clean_release
	make _clean_py
#	make _clean_c
	make _clean_bin
	make _clean_plot

docs:
	@(cd docs; make clean; make html)

ext:
	python setup.py build_ext --inplace

install:
	pip install -U -e .[dev]

release_clean:
	make clean
	-rm -r src/*.egg-info

release:
	make release_clean
	# python setup.py sdist bdist_wheel
	python setup.py sdist
	# gpg --detach-sign -a dist/gravitation*.whl
	gpg --detach-sign -a dist/gravitation*.tar.gz

.PHONY: docs
