#!/bin/bash
# Installs ScienceWorld for the scienceworld environment into ${SW} (default /home01/hanint06/sciworld; few files,
# since the group's scratch file quota is nearly full): the scienceworld 1.2.3 and py4j 0.10.9.9 wheels unpacked into
# ${SW}/pydeps (pure Python plus scienceworld.jar; only the environment's workers put it on sys.path), and a portable
# Temurin 21 JRE for aarch64 in ${SW}/jre (the NeMo-RL image has no Java). Run on the data mover (hangang-dm).
set -euo pipefail
SW=${SW:-/home01/hanint06/sciworld}
mkdir -p ${SW}/dl ${SW}/pydeps && cd ${SW}/dl
fetch() {  # url sha256
    local f=$(basename "${1%%\?*}"); [[ -f $f ]] || curl -sL -o "$f" "$1"
    echo "$2  $f" | sha256sum -c -
}
fetch https://files.pythonhosted.org/packages/a8/2a/a95edc8f02830543d782aee506eaacf1344f918fb40fa475fe5898293f8c/scienceworld-1.2.3-py3-none-any.whl \
    e9737b9d099bfaad90ef053ad6ac1d795e2063ed320efdfc0e9d9c8b5bebf419
fetch https://files.pythonhosted.org/packages/bd/db/ea0203e495be491c85af87b66e37acfd3bf756fd985f87e46fc5e3bf022c/py4j-0.10.9.9-py2.py3-none-any.whl \
    c7c26e4158defb37b0bb124933163641a2ff6e3a3913f7811b0ddbe07ed61533
[[ -f temurin21-jre-aarch64.tar.gz ]] || curl -sL -o temurin21-jre-aarch64.tar.gz \
    "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9/OpenJDK21U-jre_aarch64_linux_hotspot_21.0.13_9.tar.gz"
echo "b1734db7ade49c3f6acc6ae06233b8c2dd90e3e2f00f67000ae1c972aaa64d8c  temurin21-jre-aarch64.tar.gz" | sha256sum -c -
cd ${SW}/pydeps && unzip -qo ../dl/scienceworld-1.2.3-py3-none-any.whl && unzip -qo ../dl/py4j-0.10.9.9-py2.py3-none-any.whl
cd ${SW} && tar xzf dl/temurin21-jre-aarch64.tar.gz && ln -sfn jdk-21.0.13+9-jre jre
echo "ScienceWorld in ${SW}/pydeps, Java: $(${SW}/jre/bin/java -version 2>&1 | head -1)"
