#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP

#include <cstdio>
#include <type_traits>
#include <utility>

namespace detail {

    template <typename T, typename... Types>
    constexpr bool is_any_type_of_v = (std::is_same_v<std::decay_t<T>, std::decay_t<Types>> || ...);

    template <typename T>
    const char* get_format_specifier() {
        // list of placeholder escapes
        if constexpr (is_any_type_of_v<T, char>) {
            return "%c";
        } else if constexpr (is_any_type_of_v<T, short>) {
            return "%hi";
        } else if constexpr (is_any_type_of_v<T, int>) {
            return "%i";
        } else if constexpr (is_any_type_of_v<T, long>) {
            return "%li";
        } else if constexpr (is_any_type_of_v<T, long long>) {
            return "%lli";
        } else if constexpr (is_any_type_of_v<T, unsigned short>) {
            return "%hu";
        } else if constexpr (is_any_type_of_v<T, unsigned int>) {
            return "%u";
        } else if constexpr (is_any_type_of_v<T, unsigned long>) {
            return "%lu";
        } else if constexpr (is_any_type_of_v<T, float, double>) {
            return "%g";
        } else if constexpr (is_any_type_of_v<T, long double>) {
            return "%Lg";
        } else if constexpr (is_any_type_of_v<T, const char*, char*>) {
            return "%s";
        } else if constexpr (std::is_pointer_v<std::decay_t<T>>) {
            return "%p";
        } else {
            return "ILLEGAL_ARGUMENT_TYPE";
        }
    }

    const char* get_escape_sequence(const char c) {
        // list of escape sequences
        switch (c) {
            case '%': return "%%";
            default:  return "ILLEGAL_ARGUMENT";
        }
    }

    int c_str_size(const char* str) {
        // loop until null terminator has been found
        int size = 0;
        while(str[size] != '\0') ++size;
        return size;
    }

    int find_next_occurrence(const char* str, const char* seq, int pos = 0) {
        // calculate sizes
        const int str_size = c_str_size(str);
        const int seq_size = c_str_size(seq);

        // return if the remaining str can't contain seq
        if (pos + seq_size > str_size) return -1;

        // get next occurrence
        for (int i = pos; i <= str_size - seq_size; ++i) {
            // check whether seq starts at i
            bool contains = true;
            for (int j = 0; j < seq_size; ++j) {
                if (str[i + j] != seq[j]) contains = false;
            }
            // seq contained -> return start pos
            if (contains) return i;
        }
        // end of string -> seq not found
        return -1;
    }

    int count(const char* str, const char* seq) {
        // calculate occurrences
        const int seq_size = c_str_size(seq);
        int occurrences = 0, pos = 0;
        // as long as a next occurrence could be found increment variable
        while ((pos = find_next_occurrence(str, seq, pos)) != -1) {
            ++occurrences;
            pos += seq_size;
        }
        return occurrences;
    }

    void escape_character(char* str, const char esc) {
        int str_size = c_str_size(str);
        for (int i = 0; i < str_size; ++i) {
            // character to escape found
            if (str[i] == esc) {
                // copy all characters after i one place to the right
                for (int j = str_size; j > i; --j) {
                    str[j + 1] = str[j];
                }
                // escape character
                const char* esc_seq = get_escape_sequence(esc);
                str[i] = esc_seq[0];
                str[i + 1] = esc_seq[1];
                // increase str size and advance current position
                ++str_size;
                ++i;
            }
        }
    }

    template <typename T>
    int escape_placeholder(char* str, const char* seq, int pos = 0) {
        // get next format specifier
        pos = find_next_occurrence(str, seq, pos);
        const char* format = get_format_specifier<T>();

        // calculate sizes
        int str_size = c_str_size(str);
        const int format_size = c_str_size(format);
        const int seq_size = c_str_size(seq);

        // only right shift string if the format specifier needs more space than seq
        if (format_size != seq_size) {
            for (int i = str_size; i > pos; --i) {
                str[i + format_size - seq_size] = str[i];
            }
        }
        // replace placeholder with format specifier
        for (int i = 0; i < format_size; ++i) {
            str[pos + i] = format[i];
        }

        return pos;
    }


    template <typename... Args>
    void print(const char* msg, Args&&... args) {
        const int num_placeholders = count(msg, "{}");
        // missmatch of number of plapceholders and given values
        if (num_placeholders != sizeof...(Args)) {
            printf("WRONG NUMBER OF ARGUEMNTS!!! %s",  num_placeholders > sizeof...(Args) ? "TOO MANY PLACEHOLDERS" : "TOO MANY ARGUMENTS");
        } else {
            // create temporary msg and copy old one
            int msg_size = c_str_size(msg);
            char* substituted_msg = new char[msg_size + 1 + num_placeholders * 2 + count(msg, "%")];
            for (int i = 0; i <= msg_size; ++i) {
                substituted_msg[i] = msg[i];
            }

            // escape normale percentage signs
            escape_character(substituted_msg, '%');

            // escape placeholders
            int pos = 0;
            ((pos = escape_placeholder<Args>(substituted_msg, "{}", pos)), ...);

            // print substituted message
            std::printf(substituted_msg, std::forward<Args>(args)...);

            // delete temporary msg
            delete[] substituted_msg;
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
