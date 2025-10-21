#!/bin/sh
#Scriptname: build.sh
#Description: script to build sparcs debian packages with dpkg

if [ "$(id -u)" != 0 ]; then
    echo "DPKG build process should be performed with root privileges." 1>&2
    exit 1
fi

# Attempt to set sparcs_dir
# Resolve links: $0 may be a link
dir="$0"
# Need this for relative symlinks.
while [ -h "$dir" ] ; do
	ls=$(ls -ld "$dir")
	link=$(expr "$ls" : '.*-> \(.*\)$')
	if expr "$link" : '/.*' > /dev/null; then
		dir="$link"
	else
		dir=$(dirname "$dir")"/$link"
	fi
done
cd "$(dirname "$dir")" || exit 1 >/dev/null
sparcs_dir="$(pwd -P)"
build_dir="$sparcs_dir/build/dpkg"

# Attempt to determine the Python command
if [ -x "$sparcs_dir/.venv/bin/python" ] ; then
	python="$sparcs_dir/.venv/bin/python"
else
	python="/usr/bin/python"
fi
if [ ! -x "$python" ] ; then
	die "ERROR: Python is set to an invalid entry point: $python

Please setup a local virtual environment '.venv' or Python 3 to be available as '/usr/bin/python'."
fi

rm -rf "$build_dir"
mkdir -p "$build_dir/sparcs"

cd "$build_dir/sparcs" || exit 1 >/dev/null
cp -r "$sparcs_dir/lib/debian" "$build_dir/sparcs/"
chmod 755 "$build_dir/sparcs/debian/pre*" 2>/dev/null
chmod 755 "$build_dir/sparcs/debian/post*" 2>/dev/null
chmod 644 "$build_dir/sparcs/debian/install"
chmod 755 "$build_dir/sparcs/debian/rules"

version=$($python "$sparcs_dir/setup.py" --version)
if [ -z "$version" ]; then
    echo "Could not determine version from setup.py" 1>&2
    exit 1
elif echo "$version" | grep -q '\.dirty$'; then
	echo "Invalid determined dirty version from setup.py: $version" 1>&2
	exit 1
fi

sed -i "s/<version>/$version/g" "$build_dir/debian/changelog"
sed -i "s/<version>/$version/g" "$build_dir/debian/control"
sed -i "s/<version>/$version/g" "$build_dir/debian/postinst"

cp -r "$sparcs_dir/lib/etc" "$build_dir/sparcs/"
cp -r "$sparcs_dir/lib/usr" "$build_dir/sparcs/"

mkdir "$build_dir/sparcs/etc/sparcs"
cp "$sparcs_dir/conf/logging.default.conf" "$build_dir/sparcs/etc/sparcs/logging.conf"
cp "$sparcs_dir/conf/settings.default.conf" "$build_dir/sparcs/etc/sparcs/settings.conf"

mv "$build_dir/sparcs/etc/systemd/sparcs.service" "$build_dir/sparcs/debian/"
rmdir "$build_dir/sparcs/etc/systemd"

dpkg-buildpackage -us -uc
exit 0
