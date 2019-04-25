/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Enumeration of decoding state-machine.</summary>
	internal sealed class RunningState
	{
		internal const int Uninitialized = 0;

		internal const int BlockStart = 1;

		internal const int CompressedBlockStart = 2;

		internal const int MainLoop = 3;

		internal const int ReadMetadata = 4;

		internal const int CopyUncompressed = 5;

		internal const int InsertLoop = 6;

		internal const int CopyLoop = 7;

		internal const int CopyWrapBuffer = 8;

		internal const int Transform = 9;

		internal const int Finished = 10;

		internal const int Closed = 11;

		internal const int Write = 12;
	}
}
