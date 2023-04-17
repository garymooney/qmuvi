Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/garymooney/qmuvi/issues.

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

The best way to send feedback is to file an issue at https://github.com/garymooney/qmuvi/issues.

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

7. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions (3.10, 3.11), with tox:

```
    $ tox
```

8. Commit your changes and push your branch to GitHub:

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

9. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should pass all of the `tox` tests. Python tests will be run for versions 3.10 and 3.11, and linter tests will also be executed.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. When the github actions have been setup for the project, make sure that the tests pass for all supported Python versions by checking
   https://github.com/garymooney/qmuvi/actions

## VSCode Environment Setup (optional)

* install EditorConfig for VS Code extension to automatically read the .editorconfig file.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```
$ poetry patch # possible: major / minor / patch
$ git push
$ git push --tags
```

Github Actions will then deploy to PyPI if tests pass.