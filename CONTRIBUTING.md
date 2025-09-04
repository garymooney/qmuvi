Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/garymooney/qmuvi/issues](https://github.com/garymooney/qmuvi/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

qmuvi could always use more documentation, whether as part of the
official qmuvi docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/garymooney/qmuvi/issues](https://github.com/garymooney/qmuvi/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `qmuvi` for local development.

1. Fork the `qmuvi` repo on GitHub.
2. Clone your fork locally

```
    $ git clone git@github.com:your_name_here/qmuvi.git
```

3. Ensure [poetry](https://python-poetry.org/docs/) is installed.
4. Install dependencies and start your virtualenv:

```
    $ poetry install -E test -E doc -E dev
```

5. Create a branch for local development:

```
    $ git checkout -b name-of-your-bugfix-or-feature
```

   Now you can make your changes locally.

6. You can install qmuvi to make testing your changes easier. From the root of the local qmuvi repository, install qmuvi in editor mode using:

```
    $ pip install -e .
```

You can now play around with the scripts in the Examples directory.

8. Updating any package dependencies is done by modifying the pyproject.toml, then updating the poetry.lock file with the command
```
poetry lock --no-update
```
Then run the following to update the requirements.txt for use in docker testing
```
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

7. When you're done making changes, you can locally check that your changes pass the
   test with Python 3.10 by using tox

```
    $ tox -e py310
```

or using docker by running the "docker-build-image.bat" and "docker-run-image-test.bat" batch scripts (or the macos/linux equivalent) if that's easier.

1.  Commit your changes and push your branch to GitHub:

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

9. Submit a pull request through the GitHub website.

### VSCode Environment Setup (optional)

* Install EditorConfig for VS Code extension to automatically read the .editorconfig file.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should pass all of the `tox` tests. This includes linter, Python 3.10 and Python 3.11. Make sure that the tests pass by checking
   [https://github.com/garymooney/qmuvi/actions](https://github.com/garymooney/qmuvi/actions)
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all of the changes have been committed and tests have passed successfully.
Update the version number in the pyproject.toml file and the qmuvi \_\_init\_\_.py file.
Then build the packages:
```
$ poetry build
```

and deploy using
```
$ poetry publish --username __token__ --password PYPI_PROJECT_TOKEN
```

Github Actions will then deploy to PyPI if tests pass.
