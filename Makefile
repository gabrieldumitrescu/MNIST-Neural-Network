CXX:=g++
#CXXFLAGS:=-std=c++11 -pg -ggdb -Wall -Wextra
CXXFLAGS:=-std=c++11 -O3

LD:=g++
#LDFLAGS:=-pg
LDFLAGS:=-lSDL2
APP:=test
SRCS:=test.cc Matrix.cc Network.cc MNISTImage.cc MNISTDataset.cc util.cc
HDRS:=Matrix.h Network.h MNISTImage.h MNISTDataset.h util.h
OBJS:=$(patsubst %.cc,%.o,$(SRCS))


.PHONY: all clean

all:$(APP)

%.o:%.cc %.h
		$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP):$(OBJS) $(HDRS)
	$(LD) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -f $(APP) $(OBJS)