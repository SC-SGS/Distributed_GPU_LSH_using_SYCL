#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP

#include <cstdio>
#include <type_traits>
#include <utility>

namespace detail {

    template <typename T>
    const char* get_format_specifier() {
        using decayed_type = typename std::decay<T>::type;
        if (std::is_same<decayed_type, char>::value) {
            return "%c";
        } else if (std::is_same<decayed_type, short>::value) {
            return "%hi";
        } else if (std::is_same<decayed_type, int>::value) {
            return "%i";
        } else if (std::is_same<decayed_type, long>::value) {
            return "%li";
        } else if (std::is_same<decayed_type, long long>::value) {
            return "%lli";
        } else if (std::is_same<decayed_type, unsigned short>::value) {
            return "%hu";
        } else if (std::is_same<decayed_type, unsigned int>::value) {
            return "%u";
        } else if (std::is_same<decayed_type, unsigned long>::value) {
            return "%lu";
        } else if (std::is_same<decayed_type, float>::value || std::is_same<decayed_type, double>::value) {
            return "%g";
        } else if (std::is_same<decayed_type, long double>::value) {
            return "%Lg";
        } else if (std::is_same<decayed_type, const char*>::value || std::is_same<decayed_type, char*>::value) {
            return "%s";
        } else if (std::is_pointer<typename std::decay<T>::type>::value) {
            return "%p";
        } else {
            return "ILLEGAL_ARGUMENT_TYPE";
        }
    }

    int size(const char* str) {
        int size = 0;
        while(str[size] != '\0') { ++size; }
        return size;
    }

    int find_next_format_placeholder(const char* msg, int pos = 0) {
        if (msg[pos] == '\0') return -1;

        while (msg[pos + 1] != '\0') {
            if (msg[pos] == '{' && msg[pos + 1] == '}') {
                return pos;
            }
            ++pos;
        }
        return -1;
    }

    int count_placeholders(const char* str) {
        int occurrence = 0;
        int pos = 0;
        while((pos = find_next_format_placeholder(str, pos)) != -1) {
            ++occurrence;
            pos += 2;
        }
        return occurrence;
    }

    template <typename T>
    void replace_format_placeholder(char*& msg, int pos) {
        const char* format = get_format_specifier<T>();
        int format_size = size(format);
        char* new_msg = new char[size(msg) - 2 + format_size + 1];
        new_msg[size(new_msg)] = '\0';

        for (int i = 0; i < size(msg) - 2; ++i) {
            if (i < pos) {
                new_msg[i] = msg[i];
            } else {
                new_msg[i + format_size] = msg[i + 2];
            }
        }
        for (int i = 0; i < format_size; ++i) {
            new_msg[pos + i] = format[i];
        }
        delete[] msg;
        msg = new_msg;
    }


    template <typename T>
    void substitute_placeholders(char*& msg, T&&) {
        replace_format_placeholder<T>(msg, find_next_format_placeholder(msg));
    }

    template <typename T, typename... Args>
    void substitute_placeholders(char*& msg, T&&, Args&&... args) {
        replace_format_placeholder<T>(msg, find_next_format_placeholder(msg));
        substitute_placeholders(msg, args...);
    }


    template <typename... Args>
    void print(const char* msg, Args&&... args) {
        if (count_placeholders(msg) != sizeof...(Args)) {
            printf("WRONG NUMBER OF ARGUEMNTS!!! %s",
                   count_placeholders(msg) > sizeof...(Args) ? "TOO MANY PLACEHOLDERS" : "TOO MANY ARGUMENTS");
        } else {
            char* new_msg = new char[size(msg) + 1];
            for (int i = 0; i < size(msg); ++i) {
                new_msg[i] = msg[i];
            }
            new_msg[size(new_msg)] = '\0';

            substitute_placeholders(new_msg, args...);

            printf(new_msg, std::forward<Args>(args)...);
            delete[] new_msg;
        }
    }

    void print(const char* msg) {
        printf(msg);
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
