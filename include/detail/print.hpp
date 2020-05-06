/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 *
 * @brief Implements a custom print function using `{}` as placeholders.
 * @details Internally converts `{}` to the respective `printf` format specifiers and calls `printf`.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP

#include <cstdio>
#include <type_traits>
#include <utility>


namespace detail {

    /**
     * @brief Checks whether `T` is the same type as any of `Types`.
     * @tparam T the type to check
     * @tparam Types the types to check against
     */
    template <typename T, typename... Types>
    inline constexpr bool is_any_type_of_v = (std::is_same_v<std::decay_t<T>, std::decay_t<Types>> || ...);

    /**
     * @brief Returns the `printf` format specifier corresponding to the type `T`.
     * @tparam T the type to get the format specifier for
     * @return the `printf` format specifier
     */
    template <typename T>
    inline const char* get_format_specifier() {
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

    /**
     * @brief Returns the escape sequence for the character @p c.
     * @param[in] c the character to escape
     * @return the character escape sequence
     */
    inline const char* get_escape_sequence(const char c) {
        // list of escape sequences
        if (c == '%') {
            return "%%";
        } else {
            return "??";
        }
    }

    /**
     * @brief Calculates the size of the null-terminated string @p str.
     * @param[in] str the null-terminated string
     * @return the size of @p str (excluding the null-terminator)
     */
    inline int c_str_size(const char* str) {
        // loop until null terminator has been found
        int size = 0;
        while(str[size] != '\0') ++size;
        return size;
    }

    /**
     * @brief Find the next occurrence of @p seq in @p str starting at position @p pos.
     * @param[in] str the string (potentially) containing @p seq
     * @param[in] seq the sequence to search for
     * @param[in] pos the starting position
     * @return the position of the next occurrence of @p seq
     */
    inline int find_next_occurrence(const char* str, const char* seq, const int pos = 0) {
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

    /**
     * @brief Count the number of occurrences of @p seq in @p str.
     * @param[in] str the string (potentially) containing @p seq
     * @param[in] seq the sequence to count
     * @return the number of occurrences of @p seq
     */
    inline int count_sequence(const char* str, const char* seq) {
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

    /**
     * @brief Escape all occurrences of @p esp in @p str.
     * @param[inout] str the string (potentially) containing @p esc
     * @param[in] esc the character to escape
     */
    inline void escape_character(char* str, const char esc) {
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

    /**
     * @brief Replace the next character sequence @p seq starting at position @p pos in the string @p str.
     * @tparam T the type of the value which will be inserted at the next @p seq position
     * @param[inout] str the string (potentially) containing @p seq
     * @param[in] seq the sequence to replace
     * @param[in] pos the starting position
     * @return the position of the escaped sequence @p seq
     */
    template <typename T>
    inline int escape_placeholder_sequence(char* str, const char* seq, int pos = 0) {
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

    /**
     * @brief Print the given message @p msg after replaceing all occurences of `{}` with the corresponding `printf` format specifiers
     * based on the types of @p args.
     * @tparam Args the types to fill the placeholders
     * @param[in] msg the message (potentially) containing placeholders
     * @param[in] args the values to fill the placeholders
     */
    template <typename... Args>
    inline void print(const char* msg, Args&&... args) {
        const int num_placeholders = count_sequence(msg, "{}");
        // missmatch of number of plapceholders and given values
        if (num_placeholders != sizeof...(Args)) {
            printf("WRONG NUMBER OF ARGUEMNTS!!! %s",
                    num_placeholders > sizeof...(Args) ? "TOO MANY PLACEHOLDERS" : "TOO MANY ARGUMENTS");
        } else {
            // calculate sizes
            int msg_size = c_str_size(msg);
            int format_specifier_size = (c_str_size(get_format_specifier<Args>()) + ... + 0);
            int substituted_msg_size = msg_size - 2 * num_placeholders + format_specifier_size + count_sequence(msg, "%") + 1;
            // create temporary msg and copy old one
            char* substituted_msg = new char[substituted_msg_size];
            for (int i = 0; i <= msg_size; ++i) {
                substituted_msg[i] = msg[i];
            }

            // escape normale percentage signs
            escape_character(substituted_msg, '%');

            // escape placeholders
            int pos = 0;
            ((pos = escape_placeholder_sequence<Args>(substituted_msg, "{}", pos)), ...);

            // print substituted message
            std::printf(substituted_msg, std::forward<Args>(args)...);

            // delete temporary msg
            delete[] substituted_msg;
        }
    }

}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_PRINT_HPP
