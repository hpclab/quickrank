/*
  ---------------------------------------- 
  ParamsMap - A Simple Command Line Parser
  ----------------------------------------

MIT License

Contributors and Copyright (c):
  2016 Claudio Lucchese - claudio.lucchese@isti.cnr.it

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <unistd.h>
#include <iostream>


/// This class implements a simple command line parser
/// which also stores command line parameters in a simple data structure
/// accessible through a get method.
///
/// See Readme for an example of usage.
class ParamsMap {
public:
  ParamsMap() {};

  ~ParamsMap() {
    for (auto &x : _options)
      delete x;
  };

  /// Parse the input command line.
  ///
  /// \param argc The main's argc param
  /// \param argv The main's argv param
  /// \return True if parsing is ok, false otherwise.
  bool parse(int argc, char* argv[]) {
    for (int i=0; i<argc; i++) {
      for (auto &opt:_options) {
        Option::MATCH_STATUS m = opt->match(argv[i], i+1<argc?argv[i+1]:nullptr);
        switch (m) {
          case Option::MATCH_STATUS::MATCHED:
            // ok, skip one in argv if there was a param
            if (opt->hasArg()) i++;
            break;
          case Option::MATCH_STATUS::MISSING_ARG:
            // argument is missing
            std::cerr << "! ERROR: missing argument for "<< argv[i] << std::endl;
            return false;
          case Option::MATCH_STATUS::WRONG_ARG:
            // argument is missing
            std::cerr << "! ERROR: wrong argument format for "<< argv[i] << std::endl;
            return false;
          default:
            break;
        }
        // go to next option in argv
        if (m==Option::MATCH_STATUS::MATCHED)
          break;
      }
      // no matches with options
      _other_options.push_back(argv[i]);
    }
    return true;
  }

  /// \param  key the long-option used to register the option
  /// \return the value read from the command line, casted properly,
  ///         or the default value associated.
  template<class T>
  T get(std::string key) const {
    for (auto &opt:_options) {
      if (key==opt->getKey()) {
        if (CmdOption<T>* o = dynamic_cast<CmdOption<T>*>(opt)) {
          return o->getArg();
        }
      }
    }
    return keyNotFound<T>();
  }

  /// \param  key the long-option used to register the option
  /// \return the number of times the option was set (e.g., for options line -v -v -v)
  int count(std::string key) {
    for (auto &opt:_options) {
      if (key==opt->getKey())
        return opt->count();
    }
    return 0;
  }

  /// \param  key the long-option used to register the option
  /// \return True iff the key was set in the command line
  bool isSet(std::string key) {
    for (auto &opt:_options) {
      if (key==opt->getKey())
        return opt->count()>0;
    }
    return false;
  }

  /// \param message This is not a true option, but it is useful for a pretty help message
  void addMessage(std::string message) {
    _options.push_back(new Option(message));
  }

  /// \param long_opt The "--long-opt" name, also used as a key to retrieve the parsed value
  /// \param long_opt The "-s" short option name
  /// \param message The message added to the help message
  void addOption(std::string long_opt, std::string short_opt, std::string message) {
    _options.push_back(new CmdOption<bool>(long_opt, message, short_opt, long_opt, false, false));
  }

  /// without short-opt
  void addOption(std::string long_opt, std::string message) {
    _options.push_back(new CmdOption<bool>(long_opt, message, std::string(), long_opt, false, false));
  }

  /// \param def_val This option also accepts a param, and this is the default value provided which must be of the proper type.
  template<class T>
  void addOptionWithArg( std::string long_opt, std::string short_opt, std::string message, T def_val) {
    _options.push_back(new CmdOption<T>(long_opt, message, short_opt, long_opt, def_val, true));
  }

  /// without default value (e.g., useful for filenames)
  template<class T>
  void addOptionWithArg( std::string long_opt, std::string short_opt, std::string message) {
    _options.push_back(new CmdOption<T>(long_opt, message, short_opt, long_opt, true));
  }

  /// without short-opt
  template<class T>
  void addOptionWithArg( std::string long_opt, std::string message, T def_val) {
    _options.push_back(new CmdOption<T>(long_opt, message, std::string(), long_opt, def_val, true));
  }

  /// without short-opt and without default value
  template<class T>
  void addOptionWithArg( std::string long_opt, std::string message) {
    _options.push_back(new CmdOption<T>(long_opt, message, std::string(), long_opt, true));
  }

  /// \return A string with the help message which is obtained by chaining all the help messages
  /// of the available options in the same order as they were registered.
  std::string help() {
    std::stringstream message;
    for (auto &x : _options)
      message << cleanUp( x->help() ) << std::endl;
    return message.str();
  }

  /// Any other parameter which is not an option is appended to this list.
  std::vector<std::string> _other_options;


private:

  class Option {
   public:
    Option(std::string message) :
      _msg(message), _count (0) {}

    virtual ~Option(){}

    virtual std::string help()      const { return "\n" + _msg; }

    virtual std::string getKey()    const { return ""; }
    virtual bool hasArg()           const { return false; }
    virtual int count()             const { return _count; }

    enum MATCH_STATUS {MATCHED, NOT_MATCHED, MISSING_ARG, WRONG_ARG};
    virtual MATCH_STATUS match(char* opt, char* arg)  { return NOT_MATCHED; }

   protected:
    std::string _msg;    // message used for pretty print
    int _count;          // number of times the option was found
  };


  template<class T>
  class CmdOption : public Option {
   public:
    CmdOption(std::string key, std::string message, std::string short_opt, std::string long_opt,
              T def_val, bool has_arg) : CmdOption(key,message, short_opt, long_opt, has_arg) {
      _has_def_arg = true;
      _def_arg = _arg = def_val;
    }

    CmdOption(std::string key, std::string message, std::string short_opt, std::string long_opt, bool has_arg) :
      Option(message) {
      _key = key;
      if (!short_opt.empty())
        _short = "-"+short_opt;
      if (!long_opt.empty())
        _long = "--"+long_opt;
      _has_arg = has_arg;
      _has_def_arg = false;
    }

    virtual std::string help() const {
      int help_padding = 40;
      std::stringstream help;
      // indent
      help << "  ";
      // options
      if ( _short.empty() )
        help << _long;
      else if ( _long.empty())
        help << _short;
      else
        help << _short +","+ _long;
      // args
      if (hasArg()) {
        help << " <arg> ";
        if (hasDefArg())
          help << "("<< getDefArg()<<")";
      }
      int space_needed = std::max(0, help_padding - int(help.tellp()));
      help << std::string(space_needed, ' ') << _msg;
//      help << "\t" << _msg;

      return help.str();
    }

    virtual std::string getKey()   const { return _key; }
    virtual bool hasArg()          const { return _has_arg; }
    virtual bool hasDefArg()       const { return _has_def_arg; }
    virtual T getArg()             const { return _arg; }
    virtual T getDefArg()          const { return _def_arg; }

    virtual MATCH_STATUS match(char* opt, char* arg)  {
      if ( ( !_long.empty()  &&  _long==opt) ||
           ( !_short.empty() && _short==opt)   ) {
        // does not have arg
        if ( !hasArg() ) {
          _count++;
          return MATCHED;
        }
        // missing arg
        if (arg==nullptr)
          return MISSING_ARG;
        // everything is fine
        std::istringstream is(arg);
        T temp_arg;
        is >> temp_arg;
        if (is.fail()) {
          return WRONG_ARG;
        } else {
          _arg = temp_arg;
          _count++;
          return MATCHED;
        }
      }
      return NOT_MATCHED;
    }

   private:
    std::string _key;      // they key used for querying
    std::string _short;    // short option "-x"
    std::string _long;     // long option "-xxxxx"
    bool _has_arg;         // if the option has an argument
    T _arg;                // argument of the option "-x arg"
    bool _has_def_arg;     // if the option has a default argument
    T _def_arg;            // default argument
  };

private:

  std::string keyNotFound() const { return std::string(); }

  template<class T>
  T keyNotFound() const { return 0; }

  // trivially clean up color codes
  std::string cleanUp(std::string raw) {
    if (isatty(fileno(stdout)))
      return raw;
    std::regex e ("\033\\[\\d+m");
    std::string clean = std::regex_replace(raw, e, "");
    return clean;
  }

  std::vector<Option*> _options;
};