CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -pedantic -Wall -Wextra -Wno-long-long
# Bit-exact decoding needs every floating-point step as written: a fused multiply-add rounds differently, and hi-res
# files then decode wrongly (measured: 95970 of 96000 samples). The pragma in src/ll2.c covers clang and MSVC; GCC
# ignores it, so the flag is added whatever CFLAGS says.
override CFLAGS += -ffp-contract=off
# Lab switches (src/lab.h) are read only in a lab build: make MMX_LAB=1.
MMX_LAB ?= 0
CPPFLAGS = -Iinclude -Isrc -DMMX_LAB=$(MMX_LAB)
# The encoder uses POSIX threads for the parts of the analysis whose per-frame
# work is independent (src/threads.c). Nothing else is linked in.
PTHREAD  = -pthread
LDLIBS   = -lm

SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = build

LIB_SRCS = $(filter-out $(SRC_DIR)/main.c $(SRC_DIR)/play.c $(SRC_DIR)/play_miniaudio.c,$(wildcard $(SRC_DIR)/*.c))
LIB_OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o
# mmx play: the player and miniaudio (third_party/miniaudio) for the audio output; only the mmx program links them
PLAY_OBJS = $(OBJ_DIR)/play.o $(OBJ_DIR)/play_miniaudio.o
ifeq ($(shell uname -s),Linux)
AUDIO_LIBS = -ldl
endif

LIB_DIR  = lib
CORE_OBJS = $(filter-out $(OBJ_DIR)/cli.o $(OBJ_DIR)/commands.o,$(LIB_OBJS))

TEST_SRCS = $(wildcard tests/*.c)
TEST_BINS = $(patsubst tests/%.c,$(BIN_DIR)/%,$(TEST_SRCS))

all: $(BIN_DIR)/mmx

$(BIN_DIR)/mmx: $(LIB_OBJS) $(MAIN_OBJ) $(PLAY_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD) -o $@ $^ $(LDLIBS) $(AUDIO_LIBS)

# miniaudio's own code is not ours to warn about; play.c includes its header
$(OBJ_DIR)/play.o: CPPFLAGS += -Ithird_party/miniaudio
$(OBJ_DIR)/play_miniaudio.o: src/play_miniaudio.c src/play_miniaudio.h third_party/miniaudio/miniaudio.h | $(OBJ_DIR)
	$(CC) $(filter-out -std=% -pedantic -W%,$(CFLAGS)) -w $(PTHREAD) -Isrc -Ithird_party/miniaudio -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(PTHREAD) $(CPPFLAGS) -MMD -MP -c -o $@ $<

-include $(LIB_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(OBJ_DIR)/play.d

# The lossy encoder is switched off for now (src/cli.h); `make MMX_LOSSY=1` builds it back in. The stamp makes the two
# command-line objects rebuild whenever the setting changes.
MMX_LOSSY ?= 0
$(OBJ_DIR)/cli.o $(OBJ_DIR)/commands.o: CPPFLAGS += -DMMX_LOSSY=$(MMX_LOSSY)
$(OBJ_DIR)/cli.o $(OBJ_DIR)/commands.o: $(OBJ_DIR)/lossy_$(MMX_LOSSY).stamp
$(OBJ_DIR)/lossy_%.stamp: | $(OBJ_DIR)
	rm -f $(OBJ_DIR)/lossy_*.stamp
	touch $@
# the same for the lab switches, which reach every object
$(LIB_OBJS) $(MAIN_OBJ) $(OBJ_DIR)/play.o: $(OBJ_DIR)/lab_$(MMX_LAB).stamp
$(OBJ_DIR)/lab_%.stamp: | $(OBJ_DIR)
	rm -f $(OBJ_DIR)/lab_*.stamp
	touch $@

$(BIN_DIR)/%: tests/%.c $(LIB_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD) $(CPPFLAGS) -o $@ $< $(LIB_OBJS) $(LDLIBS)

$(BIN_DIR) $(OBJ_DIR) $(LIB_DIR):
	mkdir -p $@

$(LIB_DIR)/libminimix.a: $(CORE_OBJS) | $(LIB_DIR)
	ar rcs $@ $^

libminimix: $(LIB_DIR)/libminimix.a


# The decoder-only subset (what the foobar2000 component compiles) run against the checked-in vectors:
# lossless must match sample for sample (revision 7 and 8 vectors), lossy within 60 dB of the reference decode.
DECODER_SRCS = $(addprefix $(SRC_DIR)/, audio_buffer.c codec.c crc32.c decoder.c fft.c framecodec.c log.c lossless.c mdct.c \
               mmx_format.c mmx_io.c mmx_reader.c mmx_table.c nlms.c ll2.c ll2coder.c ll2codec.c workers.c psymodel.c rangecoder.c spectrum.c tns.c)
$(BIN_DIR)/decode_check: tools/decode_check.c $(DECODER_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD) $(CPPFLAGS) -o $@ $< $(DECODER_SRCS) $(LDLIBS)
check-decoder: $(BIN_DIR)/decode_check
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless.mmx components/foo_input_mmx/testdata/vector.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless8.mmx components/foo_input_mmx/testdata/vector.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless10.mmx components/foo_input_mmx/testdata/vector.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless10c.mmx components/foo_input_mmx/testdata/vector_clip.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless10h.mmx components/foo_input_mmx/testdata/vector24.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless10i.mmx components/foo_input_mmx/testdata/vector32.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/lossless10p.mmx components/foo_input_mmx/testdata/vector_loop.wav exact
	$(BIN_DIR)/decode_check components/foo_input_mmx/testdata/q7.mmx components/foo_input_mmx/testdata/q7.ref.wav snr

# A player's seek on a patchwork-landscape file: time until audio at the target is final, and the whole result.
$(BIN_DIR)/pld_seek: tools/pld_seek.c $(DECODER_SRCS) src/wav_writer.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -pthread -o $@ $< $(DECODER_SRCS) src/wav_writer.c $(LDLIBS)

# The same checks with an x86_64 build (on Apple silicon through Rosetta): x86 and ARM differ where C leaves results
# undefined (a NaN converted to an integer gave 0 on ARM and INT64_MIN on x86 - the MSVC CI caught it), so
# run this before a push that touches the lossless core. Also under UBSan: make check-decoder-ubsan.
VECTORS_EXACT = lossless:vector lossless8:vector lossless10:vector lossless10c:vector_clip lossless10h:vector24 lossless10i:vector32 lossless10p:vector_loop
check-decoder-x86: | $(BIN_DIR)
	$(CC) -arch x86_64 $(CFLAGS) $(PTHREAD) $(CPPFLAGS) -o $(BIN_DIR)/decode_check_x86 tools/decode_check.c $(DECODER_SRCS) $(LDLIBS)
	@for v in $(VECTORS_EXACT); do $(BIN_DIR)/decode_check_x86 components/foo_input_mmx/testdata/$${v%%:*}.mmx \
		components/foo_input_mmx/testdata/$${v##*:}.wav exact || exit 1; done
check-decoder-ubsan: | $(BIN_DIR)
	$(CC) -O1 -g -ffp-contract=off -fsanitize=undefined -fno-sanitize-recover=undefined $(CPPFLAGS) -pthread -o $(BIN_DIR)/decode_check_ubsan \
		tools/decode_check.c $(DECODER_SRCS) $(LDLIBS)
	@for v in $(VECTORS_EXACT); do $(BIN_DIR)/decode_check_ubsan components/foo_input_mmx/testdata/$${v%%:*}.mmx \
		components/foo_input_mmx/testdata/$${v##*:}.wav exact || exit 1; done


test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; $$t || exit 1; done
	@echo "All tests passed."

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

.PHONY: all test clean libminimix check-decoder check-decoder-x86 check-decoder-ubsan
