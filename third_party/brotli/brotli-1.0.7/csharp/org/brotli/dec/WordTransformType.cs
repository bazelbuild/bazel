/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Enumeration of all possible word transformations.</summary>
	/// <remarks>
	/// Enumeration of all possible word transformations.
	/// <p>There are two simple types of transforms: omit X first/last symbols, two character-case
	/// transforms and the identity transform.
	/// </remarks>
	internal sealed class WordTransformType
	{
		internal const int Identity = 0;

		internal const int OmitLast1 = 1;

		internal const int OmitLast2 = 2;

		internal const int OmitLast3 = 3;

		internal const int OmitLast4 = 4;

		internal const int OmitLast5 = 5;

		internal const int OmitLast6 = 6;

		internal const int OmitLast7 = 7;

		internal const int OmitLast8 = 8;

		internal const int OmitLast9 = 9;

		internal const int UppercaseFirst = 10;

		internal const int UppercaseAll = 11;

		internal const int OmitFirst1 = 12;

		internal const int OmitFirst2 = 13;

		internal const int OmitFirst3 = 14;

		internal const int OmitFirst4 = 15;

		internal const int OmitFirst5 = 16;

		internal const int OmitFirst6 = 17;

		internal const int OmitFirst7 = 18;

		internal const int OmitFirst8 = 19;

		internal const int OmitFirst9 = 20;

		internal static int GetOmitFirst(int type)
		{
			return type >= OmitFirst1 ? (type - OmitFirst1 + 1) : 0;
		}

		internal static int GetOmitLast(int type)
		{
			return type <= OmitLast9 ? (type - OmitLast1 + 1) : 0;
		}
	}
}
