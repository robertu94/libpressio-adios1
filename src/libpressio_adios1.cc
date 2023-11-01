#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include "adios_version.h"
#include "adios_read.h"
#include "adios.h"

namespace libpressio { namespace adios1_io_ns {

class cleanup {
  public:
    cleanup() noexcept: cleanup_fn([]{}), do_cleanup(false) {}

    template <class Function>
    cleanup(Function f) noexcept: cleanup_fn(std::forward<Function>(f)), do_cleanup(true) {}
    cleanup(cleanup&& rhs) noexcept: cleanup_fn(std::move(rhs.cleanup_fn)), do_cleanup(compat::exchange(rhs.do_cleanup, false)) {}
    cleanup(cleanup const&)=delete;
    cleanup& operator=(cleanup const&)=delete;
    cleanup& operator=(cleanup && rhs) noexcept { 
      if(&rhs == this) return *this;
      do_cleanup = compat::exchange(rhs.do_cleanup, false);
      cleanup_fn = std::move(rhs.cleanup_fn);
      return *this;
    }
    ~cleanup() { if(do_cleanup) cleanup_fn(); }

  private:
    std::function<void()> cleanup_fn;
    bool do_cleanup;
};
template<class Function>
cleanup make_cleanup(Function&& f) {
  return cleanup(std::forward<Function>(f));
}

class adios1_plugin : public libpressio_io_plugin {
  pressio_dtype to_libpressio_dtype(ADIOS_VARINFO* v) {
      switch (v->type) {
          case adios_byte:
              return pressio_int8_dtype;
          case adios_short:
              return pressio_int16_dtype;
          case adios_integer:
              return pressio_int32_dtype;
          case adios_long:
              return pressio_int64_dtype;
          case adios_unsigned_byte:
              return pressio_uint8_dtype;
          case adios_unsigned_short:
              return pressio_uint16_dtype;
          case adios_unsigned_integer:
              return pressio_uint32_dtype;
          case adios_unsigned_long:
              return pressio_uint64_dtype;
          case adios_real:
              return pressio_float_dtype;
          case adios_double:
              return pressio_double_dtype;
          default:
              throw std::runtime_error("unsupported ADIOS type");
      }
  }
  public:
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
        MPI_Comm comm = MPI_COMM_WORLD;
        ADIOS_FILE* f = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm);
        if(f==nullptr) {
            set_error(1, adios_errmsg());
            return nullptr;
        }
        auto cleanup_f = make_cleanup([f]{ adios_read_close(f);});
        int id = 0;
        bool found = false;
        for (id = 0; id < f->nvars; ++id) {
            if(f->var_namelist[id] == varname) {
                found = true;
                break;
            }
        }
        if(!found) {
            set_error(3, "variable not found");
            return nullptr;
        }
        ADIOS_VARINFO* v = adios_inq_var_byid(f, id);
        auto cleanup_v = make_cleanup([v]{ adios_free_varinfo(v);});

        if(v->ndim == 0) {
            set_error(2, "scalar values not supported");
            return nullptr;
        }
        pressio_dtype type = to_libpressio_dtype(v);
        std::vector<size_t> dims;
        for (int i = 0; i < v->ndim; ++i) {
           dims.emplace_back(v->dims[i]) ;
        }
        std::vector<uint64_t> start(dims.size());
        std::vector<uint64_t> count(dims.begin(), dims.end());
        std::reverse(dims.begin(), dims.end());

        dims.emplace_back(v->nsteps);
        auto ptr = std::make_unique<pressio_data>(pressio_data::owning(type, dims));

        ADIOS_SELECTION* sel = adios_selection_boundingbox (v->ndim, start.data(), count.data());
        adios_schedule_read_byid (f, sel, id, 0, v->nsteps, ptr->data());

        const int BLOCKING_MODE = 1;
        adios_perform_reads (f, BLOCKING_MODE);
        

        return ptr.release();
    }
  virtual int write_impl(struct pressio_data const* data) override{
        return 1;
    }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", filename);
    set(options, "adios1:filename", filename);
    set(options, "adios1:varname", varname);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &filename);
    get(options, "adios1:filename", &filename);
    get(options, "adios1:varname", &varname);
    return 0;
  }

  const char* version() const override { return ADIOS_VERSION; }
  int major_version() const override { return ADIOS_VERSION_MAJOR; }
  int minor_version() const override { return ADIOS_VERSION_MINOR; }
  int patch_version() const override { return ADIOS_VERSION_PATCH; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<adios1_plugin>(*this);
  }
  const char* prefix() const override {
    return "adios1";
  }

  private:
  std::string filename;
  std::string varname;

};

static pressio_register io_adios1_plugin(io_plugins(), "adios1", [](){ return compat::make_unique<adios1_plugin>(); });
}}

extern "C" void libpressio_register_adios1() {
}
