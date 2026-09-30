/* foobar2000 input for MiniMix (.mmx): the C decoder core does the work, this file only maps
   foobar2000's file service, info and decode calls onto it. The codec is a whole-file codec whose
   frames reference earlier frames, so the file is decoded front to back into one PCM buffer; playback
   starts as soon as the first frames are final and stays ahead of the decoder from then on. */
#include "stdafx.h"
#include <SDK/album_art_helpers.h>
extern "C" {
#include "minimix/mmx_format.h"
#include "minimix/mmx_reader.h"
#include "minimix/decoder.h"
#include "minimix/audio_buffer.h"
}
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>

// A diagnostic build (-DMMX_TRACE) logs every call foobar2000 makes, with time stamps, to /tmp/foo_input_mmx.log.
#ifdef MMX_TRACE
#include <cstdio>
#include <cstdarg>
#include <chrono>
#include <thread>
static double trace_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void trace(const char *fmt, ...) {
    static FILE *fp = fopen("/tmp/foo_input_mmx.log", "a");
    if (!fp) return;
    fprintf(fp, "%.3f [%04zx] ", trace_now(), std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xffff);
    va_list ap; va_start(ap, fmt); vfprintf(fp, fmt, ap); va_end(ap);
    fputc('\n', fp); fflush(fp);
}
struct TraceScope {
    const char *what; double t0;
    TraceScope(const char *w) : what(w), t0(trace_now()) { trace("> %s", w); }
    ~TraceScope() { trace("< %s  %.1f ms", what, (trace_now() - t0) * 1000.0); }
};
#define TRACE(...) trace(__VA_ARGS__)
#define TRACE_SCOPE(w) TraceScope trace_scope_(w)
#else
#define TRACE(...) ((void)0)
#define TRACE_SCOPE(w) ((void)0)
#endif

namespace {

// The reader asks for absolute byte ranges; foobar2000's file service answers them. Nothing may throw
// through the C reader, so a failure is remembered here and thrown after the C call returns.
struct FileSource {
    service_ptr_t<file> f;
    abort_callback *abort;
    bool aborted = false, failed = false;
    static size_t read_at(void *ctx, unsigned long long pos, void *buf, size_t n) {
        FileSource *s = (FileSource *)ctx;
        if (s->aborted || s->failed) return 0;
        try {
            s->f->seek(pos, *s->abort);
            return s->f->read(buf, n, *s->abort);
        } catch (exception_aborted &) { s->aborted = true; }
        catch (std::exception &) { s->failed = true; }
        return 0;
    }
};

const uint64_t CHUNK_FRAMES = 4096;      // PCM frames handed to foobar2000 per decode_run
const unsigned long STEP_HOPS = 32;      // decoder frames per step between abort checks (~0.74 s of audio)

// metadata keys the encoder writes for itself; everything else is a tag
bool internal_key(const std::string &k) {
    static const char *const exact[] = { "encoder", "source", "quality", "analysis", "mode", "bitrate_target", "threshold_offset_db",
                                         "bwe", "pns", "ll_shift", "ll_dropped", "nf", "is", "tilt", nullptr };
    for (int i = 0; exact[i]; i++) if (k == exact[i]) return true;
    return k.rfind("ref_", 0) == 0 || k.rfind("q_", 0) == 0 || k.rfind("t_", 0) == 0 || k.rfind("mmx_", 0) == 0;
}

// Reads the container structure through foobar2000's file service (payloads only when asked).
void read_mmx(const service_ptr_t<file> &f, const char *path, MMXFile *mmx, bool payloads, abort_callback &p_abort) {
    FileSource src; src.f = f; src.abort = &p_abort;
    MMXByteSource bs; bs.ctx = &src; bs.read_at = FileSource::read_at;
    const t_filesize size = f->get_size(p_abort);
    bs.size = size == filesize_invalid ? 0 : size;
    mmx_file_free(mmx);
    const int rc = mmx_reader_read_source(&bs, path, mmx, payloads ? 1 : 0);
    if (src.aborted) throw exception_aborted();
    if (rc != 0) { console::error("MiniMix: not a MiniMix file, or damaged"); TRACE("read_mmx FAILED rc %d", rc); throw exception_io_unsupported_format(); }
    TRACE("read_mmx ok: %lu blocks, payloads %d, cover %lu B", (unsigned long)mmx->block_count, payloads ? 1 : 0, (unsigned long)mmx->cover_len);
}

class input_mmx : public input_stubs {
public:
    input_mmx() { mmx_file_init(&m_mmx); std::memset(&m_pcm, 0, sizeof(m_pcm)); }
    ~input_mmx() { close_decoder(); mmx_audio_buffer_free(&m_pcm); mmx_file_free(&m_mmx); }

    void open(service_ptr_t<file> p_filehint, const char *p_path, t_input_open_reason p_reason, abort_callback &p_abort) {
        TRACE_SCOPE("open"); TRACE("  reason %d hint %s path %s", (int)p_reason, p_filehint.is_valid() ? "yes" : "no", p_path);
        if (p_reason == input_open_info_write) throw exception_tagging_unsupported();
        m_file = p_filehint;
        input_open_file_helper(m_file, p_path, p_reason, p_abort);
        m_path = p_path;
        read_container(false, p_abort);
    }

    void get_info(file_info &p_info, abort_callback &p_abort) {
        TRACE_SCOPE("get_info");
        const double rate = (double)m_mmx.sample_rate;
        const double length = rate > 0 ? (double)m_mmx.frame_count / rate : 0.0;
        p_info.set_length(length);
        p_info.info_set_int("samplerate", (t_int64)m_mmx.sample_rate);
        p_info.info_set_int("channels", (t_int64)m_mmx.channels);
        p_info.info_set_int("bitspersample", (t_int64)(m_mmx.source_bits ? m_mmx.source_bits : 16));
        const t_filesize size = m_file->get_size(p_abort);
        if (size != filesize_invalid && length > 0)
            p_info.info_set_bitrate((t_int64)((double)size * 8.0 / length / 1000.0 + 0.5));

        const bool lossless_codec = m_mmx.codec_id == MMX_CODEC_LOSSLESS;
        const bool bit_exact = lossless_codec && m_mmx.ll_dropped == 0;
        p_info.info_set("codec", "MiniMix");
        p_info.info_set("encoding", bit_exact ? "lossless" : "lossy");
        std::string profile;
        if (bit_exact) profile = "lossless";
        else if (lossless_codec) {
            profile = "near-lossless, " + std::to_string((int)m_mmx.ll_dropped) + " bit rounded (max ±" +
                      std::to_string(1 << (m_mmx.ll_dropped - 1)) + " LSB)";
        } else {
            const char *target = mmx_file_metadata_get(&m_mmx, "bitrate_target");
            const char *mode = mmx_file_metadata_get(&m_mmx, "mode");
            if (target) profile = std::string(target) + " kbit/s target";
            else if (mode) profile = mode;
            else profile = "quality " + std::to_string((int)m_mmx.quality);
            if (m_mmx.bwe_hz) profile += ", band replication from " + std::to_string(m_mmx.bwe_hz / 1000) + " kHz";
        }
        p_info.info_set("codec_profile", profile.c_str());
        if (const char *enc = mmx_file_metadata_get(&m_mmx, "encoder")) p_info.info_set("tool", enc);

        // tags: every metadata line that is not the encoder's own bookkeeping
        if (m_mmx.metadata) {
            std::string all(m_mmx.metadata, m_mmx.metadata_len);
            size_t pos = 0;
            while (pos < all.size()) {
                size_t nl = all.find('\n', pos); if (nl == std::string::npos) nl = all.size();
                std::string line = all.substr(pos, nl - pos); pos = nl + 1;
                size_t eq = line.find('=');
                if (eq == std::string::npos || eq == 0) continue;
                std::string key = line.substr(0, eq), value = line.substr(eq + 1);
                if (value.empty() || internal_key(key)) continue;
                p_info.meta_set(key.c_str(), value.c_str());
            }
        }
    }
    t_filestats2 get_stats2(unsigned f, abort_callback &a) { return m_file->get_stats2_(f, a); }
    t_filestats get_file_stats(abort_callback &a) { return m_file->get_stats(a); }

    void decode_initialize(unsigned p_flags, abort_callback &p_abort) {
        TRACE_SCOPE("decode_initialize"); TRACE("  flags %#x", p_flags);
        if (!m_payloads) read_container(true, p_abort);
        close_decoder();
        mmx_audio_buffer_free(&m_pcm);
        if (mmx_decoder_open(&m_mmx, &m_pcm, &m_dec) != 0) { console::error("MiniMix: unsupported stream"); throw exception_io_unsupported_format(); }
        m_pos = 0;
    }
    bool decode_run(audio_chunk &p_chunk, abort_callback &p_abort) {
        const uint64_t total = m_mmx.frame_count;
        if (m_pos >= total) { TRACE("decode_run: end of stream at %llu", (unsigned long long)m_pos); return false; }
        if ((m_pos / CHUNK_FRAMES) % 64 == 0) TRACE("decode_run: pos %llu valid %llu", (unsigned long long)m_pos, (unsigned long long)mmx_decoder_valid_frames(m_dec));
        const uint64_t want = std::min<uint64_t>(total, m_pos + CHUNK_FRAMES);
        ensure_final(m_pos, want, p_abort);
        const uint64_t avail = std::min<uint64_t>(mmx_decoder_final_from(m_dec, m_pos), want);
        if (avail <= m_pos) return false;
        const size_t n = (size_t)(avail - m_pos);
        p_chunk.set_data_32(m_pcm.samples + (size_t)m_pos * m_mmx.channels, n, m_mmx.channels, (unsigned)m_mmx.sample_rate);
        m_pos += n;
        return true;
    }
    // The same, plus the PCM as little-endian integers in the file's bit depth: what the integrity
    // verifier hashes. For a lossless file that is exactly what the encoder saw.
    bool decode_run_raw(audio_chunk &p_chunk, mem_block_container &p_raw, abort_callback &p_abort) {
        const uint64_t before = m_pos;
        if (!decode_run(p_chunk, p_abort)) return false;
        const size_t n = (size_t)(m_pos - before), ch = m_mmx.channels;
        const unsigned bits = m_mmx.source_bits ? m_mmx.source_bits : 16, bytes = bits / 8;
        const float *src = m_pcm.samples + (size_t)before * ch;
        const double scale = (double)(1LL << (bits - 1));
        const long long hi = (1LL << (bits - 1)) - 1, lo = -(1LL << (bits - 1));
        p_raw.set_size(n * ch * bytes);
        unsigned char *out = (unsigned char *)p_raw.get_ptr();
        for (size_t i = 0; i < n * ch; i++) {
            long long q = llround((double)src[i] * scale);
            if (q > hi) q = hi; else if (q < lo) q = lo;
            for (unsigned b = 0; b < bytes; b++) *out++ = (unsigned char)(q >> (8 * b));
        }
        return true;
    }
    void decode_seek(double p_seconds, abort_callback &p_abort) {
        TRACE_SCOPE("decode_seek"); TRACE("  to %.3f s", p_seconds);
        uint64_t target = audio_math::time_to_samples(p_seconds, (unsigned)m_mmx.sample_rate);
        if (target > m_mmx.frame_count) target = m_mmx.frame_count;
        // The exact spot unless reaching it costs more than about 1.5 s of decoding (a lane at about 8x realtime),
        // then an easier neighbouring entry point (coarse seeking must always work; foobar2000
        // keeps counting from the requested time, so after such a landing its display is off by the difference).
        if (target < m_mmx.frame_count) {
            const uint64_t at = mmx_decoder_seek_point(m_dec, target, (unsigned long long)(1.5 * 8.0 * m_mmx.sample_rate));
            if (at != target) TRACE("  too dear to reach: lands on the entry point at %.3f s", (double)at / m_mmx.sample_rate);
            target = at;
        }
        mmx_decoder_focus(m_dec, target);
        ensure_final(target, std::min<uint64_t>(target + 1, m_mmx.frame_count), p_abort);
        m_pos = target;
    }
    bool decode_can_seek() { return true; }
    bool decode_get_dynamic_info(file_info &, double &) { return false; }
    bool decode_get_dynamic_info_track(file_info &, double &) { return false; }
    void decode_on_idle(abort_callback &p_abort) { m_file->on_idle(p_abort); }

    void retag(const file_info &, abort_callback &) { throw exception_tagging_unsupported(); }
    void remove_tags(abort_callback &) { throw exception_tagging_unsupported(); }

    static bool g_is_our_content_type(const char *p_content_type) { return stricmp_utf8(p_content_type, "audio/x-minimix") == 0; }
    static bool g_is_our_path(const char *, const char *p_extension) { return stricmp_utf8(p_extension, "mmx") == 0; }
    static const char *g_get_name() { return "MiniMix decoder"; }
    static const GUID g_get_guid() {
        static const GUID guid = { 0xf3a5ebf0, 0xf17c, 0x4cfd, { 0xb3, 0x9d, 0xd1, 0x9c, 0x32, 0xd9, 0x88, 0x6c } };
        return guid;
    }
    // a seek into a part not decoded yet costs decoding time: let the host avoid seeks it does not need
    size_t extended_param(const GUID &type, size_t, void *, size_t) { return type == input_params::seeking_expensive ? 1 : 0; }

private:
    void read_container(bool payloads, abort_callback &p_abort) {
        m_payloads = payloads;
        read_mmx(m_file, m_path.c_str(), &m_mmx, payloads, p_abort);
    }
    // decodes until the frames [from, to) are final
    void ensure_final(uint64_t from, uint64_t to, abort_callback &p_abort) {
        while (mmx_decoder_final_from(m_dec, from) < to) {
            const int rc = mmx_decoder_step(m_dec, STEP_HOPS);
            if (rc < 0) { console::error("MiniMix: bitstream error"); throw exception_io_data(); }
            if (rc == 0) break;
            p_abort.check();
        }
    }
    void close_decoder() { if (m_dec) { mmx_decoder_close(m_dec); m_dec = nullptr; } }

    service_ptr_t<file> m_file;
    std::string m_path;
    MMXFile m_mmx;
    MMXAudioBuffer m_pcm;
    MMXDecoder *m_dec = nullptr;
    uint64_t m_pos = 0;
    bool m_payloads = false;
};

static input_singletrack_factory_t<input_mmx> g_input_mmx_factory;

// The container's embedded picture, for the album art viewer and the properties dialog.
class album_art_mmx : public album_art_extractor_instance {
public:
    album_art_mmx(const unsigned char *bytes, size_t len) { m_data = album_art_data_impl::g_create(bytes, len); }
    album_art_data_ptr query(const GUID &p_what, abort_callback &) override {
        TRACE("album_art.query %s", p_what == album_art_ids::cover_front ? "front" : "other");
        if (p_what != album_art_ids::cover_front || m_data.is_empty()) throw exception_album_art_not_found();
        return m_data;
    }
private:
    album_art_data_ptr m_data;
};

// v2 names the input this extractor belongs to, which is how the core pairs the two
class album_art_extractor_mmx : public album_art_extractor_v2 {
public:
    GUID get_guid() override { return input_mmx::g_get_guid(); }
    bool is_our_path(const char *, const char *p_extension) override { return stricmp_utf8(p_extension, "mmx") == 0; }
    album_art_extractor_instance_ptr open(file_ptr p_filehint, const char *p_path, abort_callback &p_abort) override {
        TRACE_SCOPE("album_art.open"); TRACE("  hint %s path %s", p_filehint.is_valid() ? "yes" : "no", p_path);
        file_ptr f = p_filehint;
        if (f.is_empty()) filesystem::g_open_read(f, p_path, p_abort);
        MMXFile mmx;
        mmx_file_init(&mmx);
        read_mmx(f, p_path, &mmx, false, p_abort);
        if (!mmx.cover || mmx.cover_len == 0) { mmx_file_free(&mmx); throw exception_album_art_not_found(); }
        service_ptr_t<album_art_mmx> out = new service_impl_t<album_art_mmx>(mmx.cover, (size_t)mmx.cover_len);
        mmx_file_free(&mmx);
        return out;
    }
};

static service_factory_single_t<album_art_extractor_mmx> g_album_art_extractor_mmx_factory;


}  // namespace

DECLARE_FILE_TYPE("MiniMix files", "*.mmx");
