# Build the reference C++ implementation.
#
#   make                 # both binaries, using system Boost
#   make BOOST=shim      # both binaries, using the bundled minimal Boost stand-in
#   make validate_t3     # the T3-vs-exact-DP validation harness only
#   make asan            # validation harness with AddressSanitizer + UBSan
#   make run-validate    # build and run the harness
#   make clean
#
# The code needs Boost.Graph and OpenMP.  If Boost is not installed,
# BOOST=shim uses boost_shim/, which implements only the four Boost.Graph
# facilities this file uses -- enough to build and validate, not a general
# replacement.

CXX      ?= g++
STD      ?= -std=c++17
OPT      ?= -O2
CXXFLAGS ?= $(STD) $(OPT) -fopenmp
BOOST    ?= system

ifeq ($(BOOST),shim)
INCLUDES := -Iboost_shim
else
INCLUDES :=
endif

ASANFLAGS := $(STD) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp

.PHONY: all validate_t3 solve_clsp_ib asan run-validate run-solve clean

all: solve_clsp_ib validate_t3

# the file's own main(): times the algorithm on a built-in instance set
solve_clsp_ib: solve_CLSP_IB.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# validate_T3.cpp renames the file's main() away and supplies its own
validate_t3: validate_T3.cpp solve_CLSP_IB.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

asan: validate_T3.cpp solve_CLSP_IB.cpp
	$(CXX) $(ASANFLAGS) $(INCLUDES) $< -o validate_t3_asan

run-validate: validate_t3
	./validate_t3 20 2 4 999

run-solve: solve_clsp_ib
	./solve_clsp_ib

clean:
	rm -f solve_clsp_ib validate_t3 validate_t3_asan *.o
