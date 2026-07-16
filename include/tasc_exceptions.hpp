#ifndef TASC_EXCEPTIONS_
#define TASC_EXCEPTIONS_

#include <stdexcept>
#include <string>


namespace tasc {




/**
 * Thrown to indicate that a precondition check was not met
 * 
 * Inherits from `std::exception`
 */
class assertion_error : public std::exception {
private:
    /**
     * Error message set by the user
     */
    std::string message;

public:
    /**
     * Creates a new assertion error object with the message `msg`
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
 * Thrown to indicate that a network, or a component of a network,
 * is not in the correct configuration for an action.
 * 
 * Inherits from `std::exception`
 */
class invalid_config : public std::exception {
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
    explicit invalid_config(std::string msg = "") : message(std::move(msg)) {
    }

    /**
     * @return the exception object's error message
     */
    const char* what() const noexcept override {
        return message.c_str();
    }
};



/**
 * Thrown to indicate that a method is incomplete, and thus should not be used.
 * 
 * Inherits from `std::logic_error`
 */
class not_implemented : public std::logic_error {

public:
    /**
     * Creates a new exception object with the message `msg`
     * @param msg error message to be displayed on throw
     */
    explicit not_implemented(std::string msg = "") : std::logic_error(msg) {
    }
};




/**
 * Raises `CNet::assertion_error` if `condition` is false and `NDEBUG` is not defined.
 * 
 * Does nothing if `NDEBUG` is defined.
 * 
 * @param condition boolean condition to check
 * @param assertion_fail_message message to display on assertion failure. Default: the empty string
 */
void massert(bool condition, const std::string& assertion_fail_message = "") {
    #ifndef NDEBUG 
        if(!condition) {
            throw assertion_error(assertion_fail_message);
        }
    #endif 
}



}
#endif