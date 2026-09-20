// A minimal program that needs a base relocation so that the linker emits a .reloc section.
void *self = &self;
int mainCRTStartup(void) { return self == 0; }
