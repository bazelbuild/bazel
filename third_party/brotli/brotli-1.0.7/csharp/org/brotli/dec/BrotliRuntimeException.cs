/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Unchecked exception used internally.</summary>
	[System.Serializable]
	internal class BrotliRuntimeException : System.Exception
	{
		internal BrotliRuntimeException(string message)
			: base(message)
		{
		}

		internal BrotliRuntimeException(string message, System.Exception cause)
			: base(message, cause)
		{
		}
	}
}
