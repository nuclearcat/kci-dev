#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import toml
import requests
import json
from git import Repo


def api_connection(host):
    click.secho("api connect: " + host, fg="green")
    return host


def send_build(url, patch, branch, treeurl, token):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer {}".format(token),
    }
    values = {
        "treeurl": treeurl,
        "branch": branch,
        "commit": "example",
        "kbuildname": "example",
        "testname": "example",
    }
    response = requests.post(url, headers=headers, files={"patch": patch}, data=values)
    click.secho(response.status_code, fg="green")
    click.secho(response.json(), fg="green")


def load_toml(settings):
    with open(settings) as fp:
        config = toml.load(fp)
    return config


@click.command(help="Test a patch or a mbox file")
@click.option(
    "--repository",
    default="mainline",
    help="define the kernel upstream repository where to test local changes",
)
@click.option("--branch", default="master", help="define the repository branch")
@click.option(
    "--private",
    default=False,
    is_flag=True,
    help="define if the test results will be published",
)
@click.option("--patch", required=True, help="mbox or patch file path")
@click.option("--settings", default=".kci-dev.toml", help="path of toml setting file")
def patch(repository, branch, private, patch, settings):
    config = load_toml(settings)
    url = api_connection(config["connection"]["host"])
    patch = open(patch, "rb")
    send_build(url, patch, branch, repository, config["connection"]["token"])


if __name__ == "__main__":
    main_kcidev()
