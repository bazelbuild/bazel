using System;
using System.Runtime.Remoting.Messaging;

namespace example_transitive_lib
{
    public class TransitiveClass
    {
        public static string Message
        {
            get { return "Hello World!"; }
        }
    }
}

