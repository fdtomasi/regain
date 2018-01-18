# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue (preferred),
email, or any other method with the owners of this repository before making a change. 

Please note we have a [code of conduct](https://github.com/fdtomasi/regain/blob/master/CODE_OF_CONDUCT.md),
please follow it in all your interactions with the project.

If you feel the change you are about to propose will make a useful controbution for other users too, please follow the following steps.

## How to contribute

### TL;DR
- Fork the project & clone locally.
- Create an upstream remote and sync your local copy before you branch.
- Branch for each separate piece of work.
- do some work, write good commit messages, and read the CONTRIBUTING file if there is one.
- Push to your origin repository.
- Create a new PR in GitHub.
- Respond to any code review feedback.

### Guide
#### 1. Fork the project and clone the forked copy on your computer.

Go to https://github.com/fdtomasi/regain/ and press on the "fork" button on the right.
Then, clone your forked copy with
```bash
$ git clone https://github.com/yourreponame/regain.git
```
and set up the remote
```bash
$ cd regain
$ git remote add upstream https://github.com/yourreponame/regain.git
```
Now there are two remotes for this project on your computer:

- `origin` which points to your GitHub fork of the project. You can read and write to this remote.
- `upstream` which points to the main project's GitHub repository. You can only read from this remote.

#### 2. Create a branch, and make changes.
```bash
$ git checkout master
$ git pull upstream master && git push origin master
$ git checkout -b hotfix/readme-update
```
Feel free to replace the last line according to your preference. For example, to implement a new feature,
you may name your branch with something like
```bash
$ git checkout -b feature/new-estimator
```
#### 3. Pull Request
Push the new branch with
```bash
$ git push -u origin hotfix/readme-update
```
After the changes, go to https://github.com/yourreponame/regain/, that is the Github page of your fork of the project.
To open a PR, simply press the "Compare & pull request button".

#### 4. Review
After that, we will analyse your PR and suggest some changes, before merging the PR into our project.
