%define rpmname gpucv

Summary: A GPU accelerated implementation of OpenCV library.
Name: %{rpmname}
Version: 0.4.1
Release: 175
Copyright: CeCill
Packager: Yannick Allusse, INT-Evry
Group: Development/Libraries

Source: http://
BuildRoot: /var/tmp/%{name}-buildroot
URL: https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome

%description
GpuCV is an open-source GPU-accelerated image processing and Computer Vision library. It offers an Intel's OpenCV-like programming interface for easily porting existing OpenCV applications, while taking advantage of the high level of parallelism and computing power available from recent graphics processing units (GPUs). It is distributed as free software under the CeCILL license.

%prep
%setup -q
[#%patch -p1 -b .buildroot]

%build
cd %{BuildRoot}
export RPM_OPT_FLAGS="$RPM_OPT_FLAGS"
export CONFIG=Debug
make all
export CONFIG=Release
make all


%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/usr/bin
mkdir -p $RPM_BUILD_ROOT/usr/man/man1

install -s -m 755 eject $RPM_BUILD_ROOT/usr/bin/eject
install -m 644 eject.1 $RPM_BUILD_ROOT/usr/man/man1/eject.1

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%doc README TODO COPYING ChangeLog

/usr/bin/eject
/usr/man/man1/eject.1

%changelog
-

[ Some changelog entries trimmed for brevity.  -Editor. ]