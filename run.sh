#!/usr/bin/env bash

set -x
mvn exec:java -Dexec.mainClass="test.App" -Dexec.args="$*"
