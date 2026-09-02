#!/bin/sh
set -eu

repository="t1m0thyj/discord-transcribe-bot"
workflow="build-test"
branch="main"

fail() {
    printf 'install error: %s\n' "$*" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"

os=$(uname -s 2>/dev/null) || fail "could not detect operating system"
architecture=$(uname -m 2>/dev/null) || fail "could not detect CPU architecture"

case "$os" in
    Linux) platform_os="linux" ;;
    Darwin) platform_os="macos" ;;
    CYGWIN*|MINGW*|MSYS*) platform_os="windows" ;;
    *) fail "unsupported operating system: $os" ;;
esac

case "$architecture" in
    x86_64|amd64) platform_arch="x86_64" ;;
    arm64|aarch64) platform_arch="aarch64" ;;
    *) fail "unsupported CPU architecture: $architecture" ;;
esac

case "$platform_os-$platform_arch" in
    linux-x86_64|linux-aarch64|macos-aarch64|windows-x86_64) ;;
    *) fail "no prebuilt artifact for $platform_os-$platform_arch" ;;
esac

artifact="transcribe-bot-$platform_os-$platform_arch"
download_url="https://nightly.link/$repository/workflows/$workflow/$branch/$artifact.zip"
archive=""

cleanup() {
    if [ -n "$archive" ]; then
        rm -f -- "$archive"
    fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

archive=$(mktemp "${TMPDIR:-/tmp}/transcribe-bot.XXXXXX") || fail "could not create temporary file"

printf 'Downloading latest successful %s build...\n' "$artifact"
curl -fL --retry 3 --connect-timeout 15 -o "$archive" "$download_url" \
    || fail "could not download $artifact"

if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$archive" -d .
elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$archive" -C .
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$archive" .
else
    fail "unzip, bsdtar, or python3 is required to extract the artifact"
fi

binary="./transcribe-bot"
if [ "$platform_os" = "windows" ]; then
    binary="$binary.exe"
fi

[ -f "$binary" ] || fail "artifact did not contain the expected $binary binary"
chmod +x "$binary" 2>/dev/null || true

printf 'Installed %s in %s\n' "$binary" "$PWD"
if [ "$platform_os" = "linux" ] \
    && command -v systemctl >/dev/null 2>&1 \
    && systemctl --user is-active --quiet transcribe-bot.service 2>/dev/null; then
    systemctl --user restart transcribe-bot.service
    exit
fi
printf 'To configure it, run: %s init\n' "$binary"
