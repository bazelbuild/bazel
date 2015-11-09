using System;

using NUnit.Framework;

using example_lib;

namespace example_test
{
    [TestFixture]
    public class MyTest
    {
        [Test]
        public void MyTest1()
        {
            Assert.That("foo", Is.EqualTo("Foo"));
        }

        [Test]
        public void MyTest2()
        {
            Assert.That("bar", Is.EqualTo("bar"));
        }
    }
}