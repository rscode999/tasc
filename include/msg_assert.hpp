#ifndef CNET_ASSERT_
#define CNET_ASSERT_

#include <string>


namespace CNet {




/**
 * Thrown to indicate that a precondition check was not met
 */
class assertion_error : public std::exception {
private:
    /**
     * Error message set by the user
     */
    std::string message;

public:
    /**
     * Creates a new exception object with the message `msg`
     * @param msg error message to be displayed on throw
     */
    explicit assertion_error(std::string msg = "") : message(std::move(msg)) {
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return message.c_str();
    }
};


/**
 * Raises `CNet::assertion_error` if `condition` is false and `NDEBUG` is not defined.
 * @param condition boolean condition to check
 * @param assertion_fail_message message to display on assertion failure. Default: the empty string
 */
void msg_assert(bool condition, const std::string& assertion_fail_message = "") {
    #ifndef NDEBUG 
        if(!condition) {
            throw assertion_error(assertion_fail_message);
        }
    #endif 
}



}
#endif