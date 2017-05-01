(policy.isLegalConfigurationFragment(Integer.class, ConfigurationTransition.NONE))
        .isTrue();
    // TODO(mstaib): .isFalse() when dynamic configurations care which configuration a fragment was
    // specified for
    assertThat(policy.isLegalConfigurationFragment(Integer.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(Long.class)).isTrue();
    // TODO(mstaib): .isFalse() when dynamic configurations care which configuration a fragment was
    // specified for
    assertThat(policy.isLegalConfigurationFragment(Long.class, ConfigurationTransition.NONE))
        .isTrue();
    assertThat(policy.isLegalConfigurationFragment(Long.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(String.class)).isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, ConfigurationTransition.HOST))
        .isFalse();
  }

  @Test
  public void testRequiresConfigurationFragments_MapSetsLegalityBySkylarkModuleName_NoRequires()
      throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("test_fragment"))
            .requiresHostConfigurationFragmentsBySkylarkModuleName(
                ImmutableSet.of("other_fragment"))
            .build();

    assertThat(policy.getRequiredConfigurationFragments()).isEmpty();

    assertThat(policy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.NONE))
        .isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.HOST))
        .isFalse();

    assertThat(policy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(UnknownFragment.class)).isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(
                UnknownFragment.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(
                UnknownFragment.class, ConfigurationTransition.HOST))
        .isFalse();
  }

  @Test
  public void testIncludeConfigurationFragmentsFrom_MergesWithExistingFragmentSet()
      throws Exception {
    ConfigurationFragmentPolicy basePolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("test_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class, Double.class))
            .build();
    ConfigurationFragmentPolicy addedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("other_fragment"))
            .requiresHostConfigurationFragmentsBySkylarkModuleName(
                ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Boolean.class))
            .requiresHostConfigurationFragments(ImmutableSet.<Class<?>>of(Character.class))
            .build();
    ConfigurationFragmentPolicy combinedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .includeConfigurationFragmentsFrom(basePolicy)
            .includeConfigurationFragmentsFrom(addedPolicy)
            .build();

    assertThat(combinedPolicy.getRequiredConfigurationFragments())
        .containsExactly(Integer.class, Double.class, Boolean.class, Character.class);
    assertThat(combinedPolicy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(combinedPolicy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
  }
}
