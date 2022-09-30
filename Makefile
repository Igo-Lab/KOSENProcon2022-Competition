CXX=g++
CXXFLAGS=-std=c++14
NVCC=nvcc
NVCCFLAGS= -arch=sm_53 --shared -Xcompiler -fPIC $(CXXFLAGS)
SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
BUILDDIR=build
TARGET=$(BUILDDIR)/libresolver.so


.PHONY: build
build: $(TARGET) test

$(TARGET): $(OBJS)
	[ -d $(BUILDDIR) ] || mkdir $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)

.PHONY: test
test:
	pytest -sv tests
 	