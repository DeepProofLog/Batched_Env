import janus_swi as janus
import os # Make sure os is imported if testing paths

print("Testing basic Janus query...")
try:
    # Query that doesn't require consulting a file
    result = janus.query_once("member(X, [apple, banana])")
    print(f"Result of member/2 query: {result}") # Should be {'X': 'apple'}
    if result is None or result is False:
         print("ERROR: Basic member/2 query failed!")
    else:
         print("Basic member/2 query seems to work.")

    result_true = janus.query_once("true")
    print(f"Result of 'true.' query: {result_true}") # Should be {} or True
    if result_true is None or result_true is False:
         print("ERROR: Basic 'true.' query failed!")

    result_fail = janus.query_once("fail")
    print(f"Result of 'fail.' query: {result_fail}") # Should be False or None
    if result_fail is not False and result_fail is not None:
         print("ERROR: Basic 'fail.' query did not return False/None!")

except Exception as e:
    print(f"ERROR during basic Janus query execution: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting janus.consult again interactively...")
test_file_path = './data/countries_s3/test_consult.pl' # Use absolute path
print(f"Attempting to consult: {test_file_path}")
if not os.path.exists(test_file_path):
    print(f"ERROR: Test file does not exist at path: {test_file_path}")
else:
    try:
        consult_result = janus.consult(test_file_path)
        print(f"Interactive consult result: {consult_result}") # *** We expect this to be None based on script output ***
        if consult_result is True:
             print("INTERESTING: Consult worked interactively but not in script?")
        else:
             print("CONFIRMED: Consult fails even interactively.")
    except Exception as e:
        print(f"ERROR during interactive janus.consult: {e}")
        import traceback
        traceback.print_exc()

# Exit python
exit()