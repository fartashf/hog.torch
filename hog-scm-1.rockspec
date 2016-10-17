package = "hog"
version = "scm-1"

source = {
   url = "git://github.com/fartashf/hog.torch",
   tag = "master"
}

description = {
   summary = "A Cuda implementation of HOG",
   detailed = [[
   	    A Cuda implementation of Histogram of Oriented Gradients (HOG)
   ]],
   homepage = "https://github.com/fartashf/hog.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "cutorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
]],
   install_command = "cd build"
}
