#!/bin/bash
loginfo() { echo "[INFO] $@"; }
logerror() { echo "[ERROR] $@" 1>&2; }

loginfo "==========================================================="
loginfo "开始", ${version}, "版本编译"

echo "python3 src/script.py book"
python3 src/script.py "book"

rm -rf node_modules/gitbook-plugin-tbfed-pagefooter
gitbook install

echo "python3 src/script.py powered"
python3 src/script.py "powered"

echo "python3 src/script.py gitalk"
python3 src/script.py "gitalk"

gitbook build ./ _book

# rm -rf /opt/apache-tomcat-9.0.17/webapps/test_book
# cp -r _book /opt/apache-tomcat-9.0.17/webapps/test_book
