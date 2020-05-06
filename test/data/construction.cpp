#include <data.hpp>

#include <gtest/gtest.h>

TEST(DataTest, ConstructionTest) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, 10, 3);

    EXPECT_EQ(data.size, 10);
    EXPECT_EQ(data.dims, 3);
}
