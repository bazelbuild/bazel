using System;
using System.Runtime.Remoting.Messaging;

using example_transitive_lib;

namespace example_lib
{
    public class MyClass
    {
        public string Message
        {
            get { return example_transitive_lib.TransitiveClass.Message; }
        }

        public MyClass()
        {
        }
    }
}
